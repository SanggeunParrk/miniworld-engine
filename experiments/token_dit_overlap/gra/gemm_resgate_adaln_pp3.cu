// gemm_resgate_adaln_pp3.cu -- the fused residual GEMM with three consumer warpgroups in rotation (sm_90a).
// Same contract as gemm_resgate_adaln.cu / _pp.cu:
//
//   acc = A W^T                    A [M,K] bf16 (row stride sa), W [768,K] bf16
//   x  += sigmoid(gl[tok]) * acc   x [M,768] fp32, in place
//   xa  = LN(x) * sigmoid(ms[tok]) + mb[tok]        bf16, only when ADALN
//
// Why this shape (the two-warpgroup _pp.cu measured it): a 64-row tile's epilogue on one warpgroup took about twice its
// mainloop, so two alternating warpgroups could not hide it, and with only ~2 tiles a CTA the last tile's epilogue was
// always exposed. Here a cluster of 8 CTAs splits the 768 columns (96 each), which halves a tile's epilogue and gives
// a CTA ~4 tiles at L768; three consumer warpgroups take tiles in rotation, so one is in its mainloop while two are in
// epilogues and an epilogue up to ~2x the mainloop still hides. The epilogue works from registers (x in and out, xa
// out, straight from the wgmma fragment), which leaves shared memory to a nine-stage ring.
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;

#ifndef STAGES
#define STAGES 9
#endif
#ifndef PDL
#define PDL 1
#endif
#ifdef PP_TIMES
__device__ unsigned long long g_pp[512][16][10];
TMN_DEVI unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define PST(i, k) do { if (ti == 0 && (i) < 15) g_pp[blockIdx.x][(i)][(k)] = gtimer(); } while (0)
#else
#define PST(i, k) do { } while (0)
#endif

constexpr int D_ = 768, CL = 8, NC = D_ / CL, NQ = NC / 32, KC = 64, TM = 64, NWGC = 3;
constexpr int SA = TM * 128, SB = NC * 128, SS = SA + SB;           // ring stage: A 8 KB + W 12 KB
constexpr int OST = STAGES * SS;                                    // stats [wg][parity][CL src][TM] (mean, M2)
constexpr int OMR = OST + NWGC * 2 * CL * TM * 8;                   // merged [wg][TM] (mean, rstd)
constexpr int OBAR = OMR + NWGC * TM * 8;
constexpr int SMEM_BYTES = OBAR + 256 + 1024;
constexpr uint32_t STAT_BYTES = CL * TM * 8;
static_assert(NC % 32 == 0 && (TM / CL) * 128 % 1024 == 0, "A share must be a whole swizzle atom");

TMN_DEVI float sigmoid_t(float a) {
  float t; asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * a));
  return fmaf(0.5f, t, 0.5f);
}
TMN_DEVI uint64_t dsw(uint32_t addr) { return smem_desc(addr, 16, 1024, 1); }
TMN_DEVI void mma32(float (&d)[16], uint64_t a, uint64_t b, int accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %18, 0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, p, 1, 1, 0, 0; }"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]) : "l"(a), "l"(b), "r"(accumulate));
}
TMN_DEVI void cluster_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
TMN_DEVI void cluster_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory"); }
TMN_DEVI void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
TMN_DEVI uint32_t cluster_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r)); return r; }
TMN_DEVI uint32_t cluster_id_x() { uint32_t r; asm volatile("mov.u32 %0, %%clusterid.x;\n" : "=r"(r)); return r; }
TMN_DEVI uint32_t nclusters_x() { uint32_t r; asm volatile("mov.u32 %0, %%nclusterid.x;\n" : "=r"(r)); return r; }
TMN_DEVI uint32_t mapa(uint32_t local_addr, uint32_t rank) {
  uint32_t ra; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(local_addr), "r"(rank)); return ra;
}
TMN_DEVI float2 bf2f(uint32_t u) { return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u)); }
TMN_DEVI void tma_load_2d_mc(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;"
               :: "r"(smem_u32(dst)), "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "h"(mask) : "memory");
}
TMN_DEVI void mbar_arrive_remote(uint64_t* bar, uint32_t rank) {
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(mapa(smem_u32(bar), rank)) : "memory");
}
TMN_DEVI void st_async_f2(const void* local_dst, uint64_t* local_bar, uint32_t rank, float a, float b) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];"
               :: "r"(mapa(smem_u32(local_dst), rank)), "f"(a), "f"(b), "r"(mapa(smem_u32(local_bar), rank)) : "memory");
}

template <bool ADALN>
__global__ void __launch_bounds__(128 * (NWGC + 1), 1)
grapp3_kernel(const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mw,
              const __nv_bfloat16* __restrict__ GL, const __nv_bfloat16* __restrict__ MS,
              const __nv_bfloat16* __restrict__ MB, __nv_bfloat16* __restrict__ XA, float* __restrict__ X,
              int L, int K, int ntiles, int sgl, int sms, int smb, float eps) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + OBAR);
  uint64_t* empty = full + STAGES;
  uint64_t* sbar = empty + STAGES;                                  // [wg][parity]: the cluster's row stats landed
  // [wg]: the warpgroup that had the previous tile has waited on every one of its chunks, so this one may start. A
  // waiter must never be more than one phase ahead of a barrier it shares (the parity-wait ABA found in _pp.cu).
  uint64_t* turn = sbar + NWGC * 2;
  float2* stats = reinterpret_cast<float2*>(sm + OST);
  float2* mrs = reinterpret_cast<float2*>(sm + OMR);

  const int tid = threadIdx.x, wg = tid >> 7;                       // wg 0..NWGC-1: consumers; NWGC: producer
  const uint32_t nr = cluster_rank();
  const int cid = cluster_id_x(), ncl = nclusters_x();
  const int n0 = nr * NC, nk = K / KC;
  const int nloc = cid < ntiles ? (ntiles - cid + ncl - 1) / ncl : 0;
  const int total = nloc * nk;

  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], CL * 4); }
    for (int i = 0; i < NWGC * 2; ++i) mbar_init(&sbar[i], 1);
    for (int w = 0; w < NWGC; ++w) mbar_init(&turn[w], 1);
    fence_barrier_init();
  }
  __syncthreads();
  if (ADALN && tid == 0)
    for (int i = 0; i < NWGC * 2; ++i) mbar_arrive_expect_tx(&sbar[i], STAT_BYTES);
  cluster_arrive(); cluster_wait();
#if PDL
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
  // 128 x 40 + 384 x 152 = 63488 <= 512 x 128, the launch allocation
  if (wg == NWGC) setmaxnreg_dec<40>(); else setmaxnreg_inc<152>();

  if (wg == NWGC) {                                                 // ---- producer: one thread
    if (tid == 128 * NWGC) {
      tma_prefetch_desc(&ma); tma_prefetch_desc(&mw);
      for (int i = 0; i < nloc; ++i) {
        const int m0 = (cid + i * ncl) * TM;
        for (int kc = 0; kc < nk; ++kc) {
          const int c = i * nk + kc, s = c % STAGES;
          mbar_wait(&empty[s], ((c / STAGES) & 1) ^ 1);
          mbar_arrive_expect_tx(&full[s], SS);
          tma_load_2d_mc(sm + s * SS + nr * (SA / CL), &ma, &full[s], kc * KC, m0 + nr * (TM / CL), (uint16_t)0xff);
          tma_load_2d(sm + s * SS + SA, &mw, &full[s], kc * KC, n0);
        }
      }
    }
    __syncwarp();
    cluster_arrive_relaxed(); cluster_wait();
    return;
  }

  // ---- consumers: warpgroup wg takes local tiles wg, wg + NWGC, ...
  const int lane = tid & 31, warp = (tid >> 5) & 3, ti = tid & 127;
#ifdef PP_TIMES
  if (tid == 0) g_pp[blockIdx.x][15][0] = gtimer();
#endif
  const int rr0 = warp * 16 + (lane >> 2), cb = 2 * (lane & 3);
  const uint32_t sbase = smem_u32(sm);
  for (int i = wg, j = 0; i < nloc; i += NWGC, ++j) {
    const int m0 = (cid + i * ncl) * TM, t0 = m0 % L;
    float acc[NQ][16];
    PST(i, 0);
    if (i > 0) {                                                    // k-th wait on this warpgroup's turn barrier
      const int k = wg == 0 ? j - 1 : j;
      mbar_wait(&turn[wg], k & 1);
    }
    PST(i, 1);
    for (int kc = 0; kc < nk; ++kc) {
      const int c = i * nk + kc, s = c % STAGES;
      mbar_wait(&full[s], (c / STAGES) & 1);
      const uint32_t a0 = sbase + s * SS, b0 = a0 + SA;
      wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < 4; ++ks)
#pragma unroll
        for (int q = 0; q < NQ; ++q) mma32(acc[q], dsw(a0 + ks * 32), dsw(b0 + q * 4096 + ks * 32), (kc | ks) != 0);
      wgmma_commit();
      wgmma_wait<1>();
      if (kc > 0 && c - 1 + STAGES < total && lane == 0)           // no release nobody will wait for (NODANGLE)
#pragma unroll
        for (int k = 0; k < CL; ++k) mbar_arrive_remote(&empty[(c - 1) % STAGES], k);
    }
    named_bar_sync(1 + wg, 128);
    if (ti == 0 && i + 1 < nloc) mbar_arrive(&turn[(wg + 1) % NWGC]);
    PST(i, 2);
    wgmma_wait<0>();
    PST(i, 3);
    {
      const int c = i * nk + nk - 1;
      if (c + STAGES < total && lane == 0)
#pragma unroll
        for (int k = 0; k < CL; ++k) mbar_arrive_remote(&empty[c % STAGES], k);
    }
#pragma unroll
    for (int q = 0; q < NQ; ++q) fence_regs(acc[q]);

    // ---- epilogue from registers. Fragment of m64n32: warp w, lane l holds rows 16w + l/4 (+8),
    // columns 8e + 2(l%4) + {0,1}, e = 0..3.
    const size_t ro[2] = {(size_t)(m0 + rr0) * D_ + n0 + cb, (size_t)(m0 + rr0 + 8) * D_ + n0 + cb};
    const __nv_bfloat16* gr[2] = {GL + (size_t)(t0 + rr0) * sgl + n0 + cb, GL + (size_t)(t0 + rr0 + 8) * sgl + n0 + cb};
    float2 xv[NQ][2][4]; uint32_t gv[NQ][2][4];
#pragma unroll
    for (int q = 0; q < NQ; ++q)
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          xv[q][h][e] = *reinterpret_cast<const float2*>(X + ro[h] + q * 32 + e * 8);   // x is written here: not .nc
          gv[q][h][e] = __ldg(reinterpret_cast<const unsigned int*>(gr[h] + q * 32 + e * 8));
        }
    PST(i, 4);
    float sum[2] = {0.f, 0.f};
#pragma unroll
    for (int q = 0; q < NQ; ++q)
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 g = bf2f(gv[q][h][e]);
          float& e0 = acc[q][4 * e + 2 * h];
          float& e1 = acc[q][4 * e + 2 * h + 1];
          e0 = xv[q][h][e].x + sigmoid_t(g.x) * e0;
          e1 = xv[q][h][e].y + sigmoid_t(g.y) * e1;
          sum[h] += e0 + e1;
          *reinterpret_cast<float2*>(X + ro[h] + q * 32 + e * 8) = make_float2(e0, e1);
        }
    PST(i, 5);
    if (ADALN) {
      const int par = j & 1;
      float2* st = stats + (wg * 2 + par) * CL * TM;
      uint64_t* sb = &sbar[wg * 2 + par];
      float mean[2], m2[2] = {0.f, 0.f};
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        sum[h] += __shfl_xor_sync(0xffffffffu, sum[h], 1);
        sum[h] += __shfl_xor_sync(0xffffffffu, sum[h], 2);
        mean[h] = sum[h] * (1.f / NC);
#pragma unroll
        for (int q = 0; q < NQ; ++q)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float d0 = acc[q][4 * e + 2 * h] - mean[h], d1 = acc[q][4 * e + 2 * h + 1] - mean[h];
            m2[h] += d0 * d0 + d1 * d1;
          }
        m2[h] += __shfl_xor_sync(0xffffffffu, m2[h], 1);
        m2[h] += __shfl_xor_sync(0xffffffffu, m2[h], 2);
      }
      if ((lane & 3) == 0)
#pragma unroll
        for (int k = 0; k < CL; ++k) {
          st_async_f2(&st[nr * TM + rr0], sb, k, mean[0], m2[0]);
          st_async_f2(&st[nr * TM + rr0 + 8], sb, k, mean[1], m2[1]);
        }
      const __nv_bfloat16* sr[2] = {MS + (size_t)(t0 + rr0) * sms + n0 + cb, MS + (size_t)(t0 + rr0 + 8) * sms + n0 + cb};
      const __nv_bfloat16* br[2] = {MB + (size_t)(t0 + rr0) * smb + n0 + cb, MB + (size_t)(t0 + rr0 + 8) * smb + n0 + cb};
      uint32_t sv[NQ][2][4], bv[NQ][2][4];
#pragma unroll
      for (int q = 0; q < NQ; ++q)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            sv[q][h][e] = __ldg(reinterpret_cast<const unsigned int*>(sr[h] + q * 32 + e * 8));
            bv[q][h][e] = __ldg(reinterpret_cast<const unsigned int*>(br[h] + q * 32 + e * 8));
          }
      mbar_wait(sb, (j >> 1) & 1);
      PST(i, 6);
      if ((lane & 3) == 0)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          float2 p[CL];
#pragma unroll
          for (int k = 0; k < CL; ++k) p[k] = st[k * TM + rr0 + 8 * h];
          float mu = 0.f;
#pragma unroll
          for (int k = 0; k < CL; ++k) mu += p[k].x;
          mu *= 1.f / CL;
          float M2 = 0.f;
#pragma unroll
          for (int k = 0; k < CL; ++k) { const float dm = p[k].x - mu; M2 += p[k].y + float(NC) * dm * dm; }
          mrs[wg * TM + rr0 + 8 * h] = make_float2(mu, rsqrtf(M2 * (1.f / D_) + eps));
        }
      named_bar_sync(1 + wg, 128);
      if (ti == 0) mbar_arrive_expect_tx(sb, STAT_BYTES);         // re-armed for this (wg, parity)'s next use
      const float2 mv[2] = {mrs[wg * TM + rr0], mrs[wg * TM + rr0 + 8]};
#pragma unroll
      for (int q = 0; q < NQ; ++q)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float2 sc = bf2f(sv[q][h][e]), sh = bf2f(bv[q][h][e]);
            const float o0 = (acc[q][4 * e + 2 * h] - mv[h].x) * mv[h].y * sigmoid_t(sc.x) + sh.x;
            const float o1 = (acc[q][4 * e + 2 * h + 1] - mv[h].x) * mv[h].y * sigmoid_t(sc.y) + sh.y;
            *reinterpret_cast<__nv_bfloat162*>(XA + ro[h] + q * 32 + e * 8) = __floats2bfloat162_rn(o0, o1);
          }
      named_bar_sync(1 + wg, 128);                                  // mrs is rewritten by this warpgroup's next tile
    }
    PST(i, 7);
    PST(i, 8);
  }
  __syncwarp();
  cluster_arrive_relaxed(); cluster_wait();
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
const CUtensorMap& tile_map(const torch::Tensor& t, uint32_t bi, uint32_t bo) {
  static std::map<std::array<uint64_t, 6>, CUtensorMap> cache;
  static std::mutex lock;
  const std::array<uint64_t, 6> key{reinterpret_cast<uint64_t>(t.data_ptr()), (uint64_t)t.size(0), (uint64_t)t.size(1),
                                    (uint64_t)t.stride(0), bi, bo};
  std::lock_guard<std::mutex> g(lock);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  CUtensorMap map{};
  const cuuint64_t dims[2] = {(cuuint64_t)t.size(1), (cuuint64_t)t.size(0)};
  const cuuint64_t strides[1] = {(cuuint64_t)t.stride(0) * t.element_size()};
  const cuuint32_t box[2] = {bi, bo}, elem[2] = {1, 1};
  TORCH_CHECK(encoder()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), dims, strides, box, elem,
                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed");
  return cache.emplace(key, map).first->second;
}

template <bool ADALN>
void launch(const torch::Tensor& a, const torch::Tensor& w, torch::Tensor& x, const torch::Tensor& gl,
            const torch::Tensor* ms, const torch::Tensor* mb, torch::Tensor* xa, int64_t L, double eps, int64_t max_cl) {
  const int M = a.size(0), K = a.size(1), ntiles = M / TM;
  TORCH_CHECK(K % KC == 0 && M % TM == 0 && L % TM == 0 && M % L == 0, "shape: K % 64, M % 64, L % 64");
  auto kern = grapp3_kernel<ADALN>;
  constexpr int NTHR = 128 * (NWGC + 1);
  static int maxc = [&] {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    cudaLaunchConfig_t q{};
    q.gridDim = dim3(CL * 16); q.blockDim = dim3(NTHR); q.dynamicSmemBytes = SMEM_BYTES;
    cudaLaunchAttribute a1[1];
    a1[0].id = cudaLaunchAttributeClusterDimension;
    a1[0].val.clusterDim.x = CL; a1[0].val.clusterDim.y = 1; a1[0].val.clusterDim.z = 1;
    q.attrs = a1; q.numAttrs = 1;
    int n = 0;
    TORCH_CHECK(cudaOccupancyMaxActiveClusters(&n, kern, &q) == cudaSuccess && n > 0, "no cluster fits");
    return n;
  }();
  int ncl = std::min(ntiles, maxc);
  if (max_cl > 0) ncl = std::min<int>(ncl, max_cl);
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(CL * ncl);
  cfg.blockDim = dim3(NTHR);
  cfg.dynamicSmemBytes = SMEM_BYTES;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = CL; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[1].val.programmaticStreamSerializationAllowed = PDL;
  cfg.attrs = at; cfg.numAttrs = 2;
  auto bp = [](const torch::Tensor* t) { return t ? reinterpret_cast<const __nv_bfloat16*>(t->data_ptr()) : nullptr; };
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, kern, tile_map(a, 64, TM / CL), tile_map(w, 64, NC),
                                 reinterpret_cast<const __nv_bfloat16*>(gl.data_ptr()), bp(ms), bp(mb),
                                 xa ? reinterpret_cast<__nv_bfloat16*>(xa->data_ptr()) : nullptr,
                                 reinterpret_cast<float*>(x.data_ptr()), (int)L, K, ntiles, (int)gl.stride(0),
                                 ms ? (int)ms->stride(0) : 0, mb ? (int)mb->stride(0) : 0, (float)eps) == cudaSuccess,
              "launch failed");
}
}  // namespace

void gemm_resgate_adaln_pp3(torch::Tensor a, torch::Tensor w, torch::Tensor x, torch::Tensor gl,
                            c10::optional<torch::Tensor> ms, c10::optional<torch::Tensor> mb,
                            c10::optional<torch::Tensor> xa, int64_t L, double eps, int64_t max_clusters) {
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16 && w.scalar_type() == torch::kBFloat16 && a.stride(1) == 1 && w.is_contiguous());
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 && x.is_contiguous() && x.size(1) == D_ && w.size(0) == D_);
  TORCH_CHECK(a.size(1) == w.size(1) && gl.stride(1) == 1 && gl.size(0) == L);
  const bool ad = ms.has_value();
  if (ad) TORCH_CHECK(xa->is_contiguous() && xa->scalar_type() == torch::kBFloat16 && ms->stride(1) == 1 && mb->stride(1) == 1);
  const torch::Tensor *pm = ad ? &*ms : nullptr, *pb = ad ? &*mb : nullptr;
  torch::Tensor* po = ad ? &*xa : nullptr;
  if (ad) launch<true>(a, w, x, gl, pm, pb, po, L, eps, max_clusters);
  else launch<false>(a, w, x, gl, pm, pb, po, L, eps, max_clusters);
}

torch::Tensor debug_times() {
  auto t = torch::zeros({512, 16, 10}, torch::kInt64);
#ifdef PP_TIMES
  cudaMemcpyFromSymbol(t.data_ptr(), g_pp, sizeof(g_pp));
#endif
  return t;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_resgate_adaln_pp3", &gemm_resgate_adaln_pp3);
  m.def("debug_times", &debug_times);
}
