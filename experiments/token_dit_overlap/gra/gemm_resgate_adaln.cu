// gemm_resgate_adaln.cu -- the token DiT's two residual GEMMs (attention Wo, transition squeeze) with the gated residual
// and the next half-block's AdaLN in the epilogue, sm_90a.
//
//   acc = A W^T                    A [M,K] bf16 (row stride sa), W [768,K] bf16
//   x  += sigmoid(gl[tok]) * acc   x [M,768] fp32, in place
//   xa  = LN(x) * sigmoid(ms[tok]) + mb[tok]        bf16, only when ADALN (every half-block but the last)
//
// It replaces torch.mm (y bf16 out) + resgate_adaln_rows (y in, x in/out, xa out): y never exists, and the row pass's
// launch is gone. The one thing the epilogue cannot see alone is a whole row, so a row's 768 columns are split over a
// cluster of 4 CTAs (192 each). Each CTA reduces its 192 columns to (mean, M2), publishes them in shared memory, and
// after one cluster barrier every CTA reads the other three over DSMEM and merges (Chan, equal counts).
//
// CTA: NWG consumer warpgroups (64 rows each) + 1 producer warpgroup. Mainloop is a ST-stage TMA ring of
// A [BM x 64] and W [192 x 64] tiles, 128-B swizzled; each consumer issues 3 x m64n64k16 per k-step.
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;

constexpr int D_ = 768, BN = 192, CL = 4, KC = 64, ST = 4;

TMN_DEVI float sigmoid_kit(float a) { return rcpf(__fadd_rn(1.f, ex2f(__fmul_rn(-1.4426950408889634f, a)))); }
TMN_DEVI uint64_t dsw(uint32_t addr) { return smem_desc(addr, 16, 1024, 1); }

TMN_DEVI void mma64(float (&d)[32], uint64_t a, uint64_t b, int accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 0, 0; }"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) : "l"(a), "l"(b), "r"(accumulate));
}
TMN_DEVI void cluster_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
TMN_DEVI void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
TMN_DEVI uint32_t cluster_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r)); return r; }
TMN_DEVI float2 ld_dsmem_f2(uint32_t local_addr, uint32_t rank) {
  uint32_t ra; float2 v;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(local_addr), "r"(rank));
  asm volatile("ld.shared::cluster.v2.f32 {%0,%1}, [%2];\n" : "=f"(v.x), "=f"(v.y) : "r"(ra) : "memory");
  return v;
}
TMN_DEVI float2 bf2f(uint32_t u) { return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u)); }


template <int NWG, bool ADALN>
__global__ void __launch_bounds__(128 * (NWG + 1), 1)
gra_kernel(const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mw, float* __restrict__ X,
           const __nv_bfloat16* __restrict__ GL, const __nv_bfloat16* __restrict__ MS, const __nv_bfloat16* __restrict__ MB,
           __nv_bfloat16* __restrict__ XA, int M, int L, int K, int sgl, int sms, int smb, float eps) {
  constexpr int BM = 64 * NWG, SA = BM * 128, SB = BN * 128, SS = SA + SB;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + ST * SS);
  uint64_t* empty = full + ST;
  float2* stats = reinterpret_cast<float2*>(empty + ST);          // [BM] local (mean, M2) of this CTA's 192 columns

  const int tid = threadIdx.x, wg = tid >> 7;
  const uint32_t rank = cluster_rank();
  const int n0 = rank * BN, m0 = (blockIdx.x / CL) * BM, nk = K / KC;

  if (tid == 0) {
    for (int s = 0; s < ST; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], 4 * NWG); }
    fence_barrier_init();
  }
  __syncthreads();

  if (wg == NWG) {                                                  // producer warpgroup: one thread feeds the ring
    if (tid == 128 * NWG) {
      tma_prefetch_desc(&ma); tma_prefetch_desc(&mw);
      for (int c = 0; c < nk; ++c) {
        const int s = c % ST;
        mbar_wait(&empty[s], ((c / ST) & 1) ^ 1);
        mbar_arrive_expect_tx(&full[s], SS);
        tma_load_2d(sm + s * SS, &ma, &full[s], c * KC, m0);
        tma_load_2d(sm + s * SS + SA, &mw, &full[s], c * KC, n0);
      }
    }
    __syncwarp();
    if (ADALN) { cluster_arrive(); cluster_wait(); cluster_arrive(); cluster_wait(); }
    return;
  }

  float acc[3][32];
#pragma unroll
  for (int j = 0; j < 3; ++j)
#pragma unroll
    for (int i = 0; i < 32; ++i) acc[j][i] = 0.f;

  const uint32_t sbase = smem_u32(sm);
  for (int c = 0; c < nk; ++c) {
    const int s = c % ST;
    mbar_wait(&full[s], (c / ST) & 1);
    const uint32_t a0 = sbase + s * SS + wg * 8192, b0 = sbase + s * SS + SA;
    wgmma_fence();
#pragma unroll
    for (int ks = 0; ks < 4; ++ks)
#pragma unroll
      for (int j = 0; j < 3; ++j) mma64(acc[j], dsw(a0 + ks * 32), dsw(b0 + j * 8192 + ks * 32), (c | ks) != 0);
    wgmma_commit();
    wgmma_wait<1>();
    if (c > 0 && (tid & 31) == 0) mbar_arrive(&empty[(c - 1) % ST]);
  }
  wgmma_wait<0>();
#pragma unroll
  for (int j = 0; j < 3; ++j) fence_regs(acc[j]);

  // ---- epilogue. Fragment of m64nN: warp w, lane l owns rows 16w + l/4 (+8), columns 8q + 2(l%4) + {0,1}.
  const int lane = tid & 31, warp = (tid >> 5) & 3;
  const int rl0 = wg * 64 + warp * 16 + (lane >> 2);               // row within the tile
  const int r[2] = {m0 + rl0, m0 + rl0 + 8};
  const int cb = n0 + 2 * (lane & 3);
  float sum[2] = {0.f, 0.f};
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int row = r[h] < M ? r[h] : M - 1;
    const int tok = row % L;
    float* xr = X + (size_t)row * D_;
    const __nv_bfloat16* gr = GL + (size_t)tok * sgl;
#pragma unroll
    for (int j = 0; j < 3; ++j)
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const int col = cb + j * 64 + q * 8;
        const float2 xv = *reinterpret_cast<const float2*>(xr + col);
        const float2 g = bf2f(*reinterpret_cast<const uint32_t*>(gr + col));
        float& e0 = acc[j][4 * q + 2 * h];
        float& e1 = acc[j][4 * q + 2 * h + 1];
        e0 = xv.x + sigmoid_kit(g.x) * e0;
        e1 = xv.y + sigmoid_kit(g.y) * e1;
        sum[h] += e0 + e1;
        if (r[h] < M) *reinterpret_cast<float2*>(xr + col) = make_float2(e0, e1);
      }
  }
  if (!ADALN) return;

  float mean[2], m2[2] = {0.f, 0.f};
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    sum[h] += __shfl_xor_sync(0xffffffffu, sum[h], 1);
    sum[h] += __shfl_xor_sync(0xffffffffu, sum[h], 2);
    mean[h] = sum[h] * (1.f / BN);
#pragma unroll
    for (int j = 0; j < 3; ++j)
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const float d0 = acc[j][4 * q + 2 * h] - mean[h], d1 = acc[j][4 * q + 2 * h + 1] - mean[h];
        m2[h] += d0 * d0 + d1 * d1;
      }
    m2[h] += __shfl_xor_sync(0xffffffffu, m2[h], 1);
    m2[h] += __shfl_xor_sync(0xffffffffu, m2[h], 2);
  }
  if ((lane & 3) == 0) { stats[rl0] = make_float2(mean[0], m2[0]); stats[rl0 + 8] = make_float2(mean[1], m2[1]); }
  cluster_arrive(); cluster_wait();

  float rstd[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint32_t la = smem_u32(&stats[rl0 + 8 * h]);
    float2 p[CL];
#pragma unroll
    for (int k = 0; k < CL; ++k) p[k] = ld_dsmem_f2(la, k);
    float mu = 0.f;
#pragma unroll
    for (int k = 0; k < CL; ++k) mu += p[k].x;
    mu *= 1.f / CL;
    float M2 = 0.f;
#pragma unroll
    for (int k = 0; k < CL; ++k) { const float dm = p[k].x - mu; M2 += p[k].y + float(BN) * dm * dm; }
    mean[h] = mu;
    rstd[h] = rsqrtf(M2 * (1.f / D_) + eps);
  }
  cluster_arrive();                                                 // our reads are done; peers may exit after the wait

#pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (r[h] >= M) continue;
    const int tok = r[h] % L;
    const __nv_bfloat16* sr = MS + (size_t)tok * sms;
    const __nv_bfloat16* br = MB + (size_t)tok * smb;
    __nv_bfloat16* orow = XA + (size_t)r[h] * D_;
#pragma unroll
    for (int j = 0; j < 3; ++j)
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const int col = cb + j * 64 + q * 8;
        const float2 s = bf2f(*reinterpret_cast<const uint32_t*>(sr + col));
        const float2 b = bf2f(*reinterpret_cast<const uint32_t*>(br + col));
        const float o0 = (acc[j][4 * q + 2 * h] - mean[h]) * rstd[h] * sigmoid_kit(s.x) + b.x;
        const float o1 = (acc[j][4 * q + 2 * h + 1] - mean[h]) * rstd[h] * sigmoid_kit(s.y) + b.y;
        *reinterpret_cast<__nv_bfloat162*>(orow + col) = __floats2bfloat162_rn(o0, o1);
      }
  }
  __syncwarp();
  cluster_wait();
}

// ------------------------------------------------------------------------------------------------ host
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <array>
#include <map>
#include <mutex>

namespace {
using EncodeTiled = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                 const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                 CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
EncodeTiled encoder() {
  static EncodeTiled fn = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
    TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q) == cudaSuccess && p, "no TMA");
    return reinterpret_cast<EncodeTiled>(p);
  }();
  return fn;
}
// 2-D bf16 map over a row-strided matrix [rows, cols] (cols contiguous), box [bo rows, 64 cols], 128-B swizzle.
// Cached on (pointer, shape, stride, box): a descriptor is valid for one base pointer only.
const CUtensorMap& tile_map(const torch::Tensor& t, uint32_t bo) {
  static std::map<std::array<uint64_t, 5>, CUtensorMap> cache;
  static std::mutex lock;
  const std::array<uint64_t, 5> key{reinterpret_cast<uint64_t>(t.data_ptr()), (uint64_t)t.size(0), (uint64_t)t.size(1),
                                    (uint64_t)t.stride(0), bo};
  std::lock_guard<std::mutex> g(lock);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  CUtensorMap map{};
  const cuuint64_t dims[2] = {(cuuint64_t)t.size(1), (cuuint64_t)t.size(0)};
  const cuuint64_t strides[1] = {(cuuint64_t)t.stride(0) * 2};
  const cuuint32_t box[2] = {64, bo}, elem[2] = {1, 1};
  TORCH_CHECK(encoder()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), dims, strides, box, elem,
                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed");
  return cache.emplace(key, map).first->second;
}

template <int NWG, bool ADALN>
void launch(const torch::Tensor& a, const torch::Tensor& w, torch::Tensor& x, const torch::Tensor& gl,
            const torch::Tensor* ms, const torch::Tensor* mb, torch::Tensor* xa, int64_t L, double eps) {
  constexpr int BM = 64 * NWG;
  const int M = a.size(0), K = a.size(1);
  const size_t smem = 1024 + ST * (BM * 128 + BN * 128) + 2 * ST * 8 + BM * 8;
  auto kern = gra_kernel<NWG, ADALN>;
  static bool attr = [&] { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); return true; }();
  (void)attr;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(CL * ((M + BM - 1) / BM));
  cfg.blockDim = dim3(128 * (NWG + 1));
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = CL; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  auto bp = [](const torch::Tensor* t) { return t ? reinterpret_cast<const __nv_bfloat16*>(t->data_ptr()) : nullptr; };
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, kern, tile_map(a, BM), tile_map(w, BN), x.data_ptr<float>(),
                                 reinterpret_cast<const __nv_bfloat16*>(gl.data_ptr()), bp(ms), bp(mb),
                                 xa ? reinterpret_cast<__nv_bfloat16*>(xa->data_ptr()) : nullptr, M, (int)L, K,
                                 (int)gl.stride(0), ms ? (int)ms->stride(0) : 0, mb ? (int)mb->stride(0) : 0,
                                 (float)eps) == cudaSuccess, "launch failed");
}
}  // namespace

// x += sigmoid(gl) * (a @ w^T); then, when ms is given, xa = AdaLN(x). nwg: consumer warpgroups (tile rows = 64 nwg).
void gemm_resgate_adaln(torch::Tensor a, torch::Tensor w, torch::Tensor x, torch::Tensor gl,
                        c10::optional<torch::Tensor> ms, c10::optional<torch::Tensor> mb, c10::optional<torch::Tensor> xa,
                        int64_t L, double eps, int64_t nwg) {
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16 && w.scalar_type() == torch::kBFloat16 && a.stride(1) == 1 && w.is_contiguous());
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 && x.is_contiguous() && x.size(1) == D_ && w.size(0) == D_);
  TORCH_CHECK(a.size(1) == w.size(1) && a.size(1) % KC == 0 && gl.stride(1) == 1);
  const bool ad = ms.has_value();
  const torch::Tensor *pm = ad ? &*ms : nullptr, *pb = ad ? &*mb : nullptr;
  torch::Tensor* po = ad ? &*xa : nullptr;
  if (nwg == 1) { ad ? launch<1, true>(a, w, x, gl, pm, pb, po, L, eps) : launch<1, false>(a, w, x, gl, pm, pb, po, L, eps); }
  else { ad ? launch<2, true>(a, w, x, gl, pm, pb, po, L, eps) : launch<2, false>(a, w, x, gl, pm, pb, po, L, eps); }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gemm_resgate_adaln", &gemm_resgate_adaln); }
