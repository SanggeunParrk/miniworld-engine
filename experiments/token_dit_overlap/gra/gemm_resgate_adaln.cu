// gemm_resgate_adaln.cu -- the token DiT's two residual GEMMs (attention Wo, transition squeeze) with the gated residual
// and the next half-block's AdaLN in the epilogue, sm_90a.
//
//   acc = A W^T                    A [M,K] bf16 (row stride sa), W [768,K] bf16
//   x  += sigmoid(gl[tok]) * acc   x [M,768] fp32, in place
//   xa  = LN(x) * sigmoid(ms[tok]) + mb[tok]        bf16, only when ADALN (every half-block but the last)
//
// It replaces torch.mm (y bf16 out) + resgate_adaln_rows (y in, x in/out, xa out): y never exists, and the row pass's
// launch is gone. A row's 768 columns are split over a cluster of 4 CTAs (192 each). Each CTA reduces its columns to
// (mean, M2) and pushes them into all four CTAs' shared memory with st.async (mbarrier complete_tx); every CTA then merges
// the four partials from its own shared memory (Chan, equal counts).
//
// CTA: NWG consumer warpgroups (64 rows each) + 1 producer warpgroup. Mainloop: an ST-stage TMA ring of A [BM x 64] and
// W [192 x 64] tiles, 128-B swizzled, A multicast over the cluster (each CTA loads a quarter of the rows); each consumer
// issues 3 x m64n64k16 per k-step. The gl tile is loaded before the mainloop into its own region; the x tile aliases the
// ring and is loaded slot by slot as the last chunks free it. x goes back box by box and xa goes out by TMA store.
//
// History (README): v1 26057bda plain loads after the mainloop; v3 b3b56d37 ring aliasing + multicast (mainloop at cuBLAS
// parity). v4: no cluster barrier.release in the epilogue (it waited for outstanding memory operations, 2 us), ms/mb
// loads issued before the x store stream starts, shared-memory loads batched so they do not serialize on aliasing.
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;

#ifndef STAGES
#define STAGES 4
#endif
#ifdef GRA_TIMES
__device__ unsigned long long g_times[1024][16];
TMN_DEVI unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define TSTAMP(k) do { if (threadIdx.x == 0) g_times[blockIdx.x][k] = gtimer(); } while (0)
#else
#define TSTAMP(k) do { } while (0)
#endif
constexpr int D_ = 768, BN = 192, CL = 4, KC = 64;
constexpr int XB = BN / 32, GB = BN / 64, TILE = 8192;             // per warpgroup: 6 x [64][32] fp32, 3 x [64][64] bf16

// sigmoid(a) = 0.5 tanh(a / 2) + 0.5: one MUFU op (tanh.approx, max rel err ~2^-11) instead of ex2 + rcp
TMN_DEVI float sigmoid_t(float a) {
  float t; asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * a));
  return fmaf(0.5f, t, 0.5f);
}
TMN_DEVI uint64_t dsw(uint32_t addr) { return smem_desc(addr, 16, 1024, 1); }

TMN_DEVI void mma64(float (&d)[32], uint64_t a, uint64_t b, int accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 0, 0; }"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) : "l"(a), "l"(b), "r"(accumulate));
}
TMN_DEVI void cluster_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
TMN_DEVI void cluster_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory"); }
TMN_DEVI void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
TMN_DEVI uint32_t cluster_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r)); return r; }
TMN_DEVI uint32_t mapa(uint32_t local_addr, uint32_t rank) {
  uint32_t ra; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(local_addr), "r"(rank)); return ra;
}
TMN_DEVI float2 bf2f(uint32_t u) { return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u)); }
TMN_DEVI uint32_t ldg_nc(const void* p) { uint32_t v; asm volatile("ld.global.nc.b32 %0, [%1];" : "=r"(v) : "l"(p)); return v; }
TMN_DEVI void tma_store_2d(const CUtensorMap* map, const void* src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
               :: "l"(map), "r"(smem_u32(src)), "r"(c0), "r"(c1) : "memory");
}
TMN_DEVI void tma_load_2d_mc(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;"
               :: "r"(smem_u32(dst)), "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "h"(mask) : "memory");
}
// Plain remote arrive. mbarrier.arrive.release.cluster made the mainloop 4x slower: the release waits for the
// in-flight wgmma's shared-memory reads.
TMN_DEVI void mbar_arrive_remote(uint64_t* bar, uint32_t rank) {
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(mapa(smem_u32(bar), rank)) : "memory");
}
// 8 bytes into CTA `rank`'s shared memory at the same offset, counted on its mbarrier at the same offset (complete_tx)
TMN_DEVI void st_async_f2(const void* local_dst, uint64_t* local_bar, uint32_t rank, float a, float b) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];"
               :: "r"(mapa(smem_u32(local_dst), rank)), "f"(a), "f"(b), "r"(mapa(smem_u32(local_bar), rank)) : "memory");
}
// byte offset of (row, col) in a [64][128 B] tile with the 128-B swizzle (16-B granule ^= row % 8)
TMN_DEVI uint32_t sw128(int row, int byte) { return row * 128 + ((((byte >> 4) ^ (row & 7))) << 4) + (byte & 15); }

template <int NWG> struct Cfg {
  static constexpr int BM = 64 * NWG, ST = STAGES;
  static constexpr int SA = BM * 128, SB = BN * 128, SS = SA + SB;
  // x tiles alias the stage ring (the producer loads them as the last chunks free it); gl, then xa, has its own region
  static constexpr int NT = NWG * XB, OX = 0, OG = ST * SS, OBAR = OG + NWG * GB * TILE;
  // stats [CL source CTAs][BM] (mean, M2), then the merged [BM] (mean, rstd)
  static constexpr int OST = OBAR + 128, OMR = OST + CL * BM * 8, SMEM = OMR + BM * 8 + 1024;
  static_assert(SS % TILE == 0 && NT * TILE <= (ST - 1) * SS, "x tiles must fit the ring's first ST-1 slots");
};

template <int NWG, bool ADALN>
__global__ void __launch_bounds__(128 * (NWG + 1), 1)
gra_kernel(const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mw,
           const __grid_constant__ CUtensorMap mx, const __grid_constant__ CUtensorMap mg,
           const __grid_constant__ CUtensorMap mxa, const __nv_bfloat16* __restrict__ MS,
           const __nv_bfloat16* __restrict__ MB, int L, int K, int sms, int smb, float eps) {
  using C = Cfg<NWG>;
  constexpr int ST = C::ST, SA = C::SA, SS = C::SS, BM = C::BM;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + C::OBAR);
  uint64_t* empty = full + ST;
  uint64_t* xbar = empty + ST;
  uint64_t* sbar = xbar + 1;                                        // the four CTAs' row statistics have landed
  float2* stats = reinterpret_cast<float2*>(sm + C::OST);
  float2* mrs = reinterpret_cast<float2*>(sm + C::OMR);

  const int tid = threadIdx.x, wg = tid >> 7;
  const uint32_t rank = cluster_rank();
  const int n0 = rank * BN, m0 = (blockIdx.x / CL) * BM, nk = K / KC, t0 = m0 % L;

  if (tid == 0) {
    for (int s = 0; s < ST; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], CL * 4 * NWG); }   // every consumer warp of the cluster frees a slot
    mbar_init(xbar, 1);
    mbar_init(sbar, 1);
    if (ADALN) mbar_arrive_expect_tx(sbar, CL * BM * 8);
    fence_barrier_init();
  }
  __syncthreads();
  cluster_arrive(); cluster_wait();                                 // peers' barriers exist before anyone multicasts into them
  TSTAMP(1);

  if (wg == NWG) {                                                  // producer warpgroup: one thread feeds everything
    if (tid == 128 * NWG) {
      tma_prefetch_desc(&ma); tma_prefetch_desc(&mw);
      mbar_arrive_expect_tx(xbar, NWG * (XB + GB) * TILE);
      for (int t = 0; t < NWG * GB; ++t) tma_load_2d(sm + C::OG + t * TILE, &mg, xbar, n0 + 64 * (t % GB), t0 + 64 * (t / GB));
      for (int c = 0; c < nk; ++c) {
        const int s = c % ST;
        mbar_wait(&empty[s], ((c / ST) & 1) ^ 1);
        mbar_arrive_expect_tx(&full[s], SS);
        // A is the same for the whole cluster: each CTA loads a quarter of the rows and multicasts it to all four
        tma_load_2d_mc(sm + s * SS + rank * (SA / CL), &ma, &full[s], c * KC, m0 + rank * (BM / CL), (1u << CL) - 1);
        tma_load_2d(sm + s * SS + SA, &mw, &full[s], c * KC, n0);
      }
      // x tiles into the ring, slot by slot as the last chunks free it (tile t at byte t * TILE)
      for (int e = 0; e < ST - 1; ++e) {
        const int cc = nk - ST + e, s = cc % ST;
        mbar_wait(&empty[s], (((cc + ST) / ST) & 1) ^ 1);
        for (int t = s * (SS / TILE); t < (s + 1) * (SS / TILE) && t < C::NT; ++t)
          tma_load_2d(sm + t * TILE, &mx, xbar, n0 + 32 * (t % XB), m0 + 64 * (t / XB));
      }
    }
    __syncwarp();
    cluster_arrive_relaxed(); cluster_wait();
    return;
  }

  float acc[3][32];
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
    if (c > 0 && (tid & 31) == 0)
#pragma unroll
      for (int k = 0; k < CL; ++k) mbar_arrive_remote(&empty[(c - 1) % ST], k);
  }
  wgmma_wait<0>();
  if ((tid & 31) == 0)                                              // the last chunk's slot too
#pragma unroll
    for (int k = 0; k < CL; ++k) mbar_arrive_remote(&empty[(nk - 1) % ST], k);
#pragma unroll
  for (int j = 0; j < 3; ++j) fence_regs(acc[j]);
  // Teardown barrier, arrived now and waited at the very end: peers may still arrive on our empty barriers. Relaxed:
  // it orders nothing, and a release here would wait for outstanding memory operations.
  cluster_arrive_relaxed();
  TSTAMP(2);

  // ---- epilogue. Fragment of m64nN: warp w, lane l owns rows 16w + l/4 (+8), columns 8q + 2(l%4) + {0,1}.
  const int lane = tid & 31, warp = (tid >> 5) & 3;
  const int rr0 = warp * 16 + (lane >> 2);                          // row within this warpgroup's 64
  const int rl0 = wg * 64 + rr0;                                    // row within the CTA tile
  const int cb = 2 * (lane & 3);
  uint8_t* smx = sm + C::OX + wg * XB * TILE;
  uint8_t* smg = sm + C::OG + wg * GB * TILE;
  mbar_wait(xbar, 0);
  TSTAMP(3);

  // x += sigmoid(gl) acc, one 64-column block at a time: loads batched ahead of the stores (the compiler cannot prove
  // they do not alias), then the block's two x boxes go out while the next block computes.
  float sum[2] = {0.f, 0.f};
#pragma unroll
  for (int j = 0; j < 3; ++j) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int rr = rr0 + 8 * h;
      float2 xv[8]; uint32_t gv[8];
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const int col = j * 64 + q * 8 + cb;
        xv[q] = *reinterpret_cast<const float2*>(smx + (col >> 5) * TILE + sw128(rr, (col & 31) * 4));
        gv[q] = *reinterpret_cast<const uint32_t*>(smg + j * TILE + sw128(rr, (col & 63) * 2));
      }
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const float2 g = bf2f(gv[q]);
        float& e0 = acc[j][4 * q + 2 * h];
        float& e1 = acc[j][4 * q + 2 * h + 1];
        e0 = xv[q].x + sigmoid_t(g.x) * e0;
        e1 = xv[q].y + sigmoid_t(g.y) * e1;
        sum[h] += e0 + e1;
      }
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const int col = j * 64 + q * 8 + cb;
        *reinterpret_cast<float2*>(smx + (col >> 5) * TILE + sw128(rr, (col & 31) * 4)) =
            make_float2(acc[j][4 * q + 2 * h], acc[j][4 * q + 2 * h + 1]);
      }
    }
    fence_proxy_async();
    named_bar_sync(1 + wg, 128);
    if ((tid & 127) == 0) {
      for (int b = 2 * j; b < 2 * j + 2; ++b) tma_store_2d(&mx, smx + b * TILE, n0 + 32 * b, m0 + 64 * wg);
      tma_store_commit();
    }
  }
  TSTAMP(4);
  if (!ADALN) {
    if ((tid & 127) == 0) tma_store_wait_read<0>();
    __syncwarp();
    cluster_wait();
    return;
  }

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
  if ((lane & 3) == 0)                                              // stats[rank][row] in every CTA of the cluster
#pragma unroll
    for (int k = 0; k < CL; ++k) {
      st_async_f2(&stats[rank * BM + rl0], sbar, k, mean[0], m2[0]);
      st_async_f2(&stats[rank * BM + rl0 + 8], sbar, k, mean[1], m2[1]);
    }
  TSTAMP(7);

  // The AdaLN phase reads xn back from shared memory, so its thread <-> column map is free: each thread takes 8-column
  // chunks (16-B loads of ms / mb, coalesced along the row) instead of the wgmma fragment's 2-column pairs spread over
  // 8 rows (4-B loads, 8 sectors per instruction: issuing them took 3 us). Chunk c of this warpgroup: row c / 24, col 8 (c % 24).
  constexpr int NCH = 64 * (BN / 8) / 128;                          // 12 chunks per thread
  const int ti = tid & 127;
  uint4 msv[NCH], mbv[NCH];
#pragma unroll
  for (int i = 0; i < NCH; ++i) {
    const int c = ti + 128 * i, r = c / 24, col = 8 * (c % 24);
    const size_t tok = t0 + wg * 64 + r;
    msv[i] = __ldg(reinterpret_cast<const uint4*>(MS + tok * sms + n0 + col));
    mbv[i] = __ldg(reinterpret_cast<const uint4*>(MB + tok * smb + n0 + col));
  }
  TSTAMP(9);
  mbar_wait(sbar, 0);
  TSTAMP(5);

  if ((lane & 3) == 0)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float2 p[CL];
#pragma unroll
      for (int k = 0; k < CL; ++k) p[k] = stats[k * BM + rl0 + 8 * h];
      float mu = 0.f;
#pragma unroll
      for (int k = 0; k < CL; ++k) mu += p[k].x;
      mu *= 1.f / CL;
      float M2 = 0.f;
#pragma unroll
      for (int k = 0; k < CL; ++k) { const float dm = p[k].x - mu; M2 += p[k].y + float(BN) * dm * dm; }
      mrs[rl0 + 8 * h] = make_float2(mu, rsqrtf(M2 * (1.f / D_) + eps));
    }
  named_bar_sync(1 + wg, 128);
  TSTAMP(10);

  // xa over the gl tiles (gl is fully consumed: the last resgate block ended on this warpgroup's barrier)
#pragma unroll
  for (int i = 0; i < NCH; ++i) {
    const int c = ti + 128 * i, r = c / 24, col = 8 * (c % 24);
    const float2 mr = mrs[wg * 64 + r];
    const int g = (col & 31) >> 2;                                  // first 16-B granule of the chunk in its fp32 box
    const uint8_t* xrow = smx + (col >> 5) * TILE + r * 128;
    const float4 x0 = *reinterpret_cast<const float4*>(xrow + ((g ^ (r & 7)) << 4));
    const float4 x1 = *reinterpret_cast<const float4*>(xrow + (((g + 1) ^ (r & 7)) << 4));
    const float xs[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
    const uint32_t sw[4] = {msv[i].x, msv[i].y, msv[i].z, msv[i].w}, bw[4] = {mbv[i].x, mbv[i].y, mbv[i].z, mbv[i].w};
    uint32_t o[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 sc = bf2f(sw[e]), sh = bf2f(bw[e]);
      const float o0 = (xs[2 * e] - mr.x) * mr.y * sigmoid_t(sc.x) + sh.x;
      const float o1 = (xs[2 * e + 1] - mr.x) * mr.y * sigmoid_t(sc.y) + sh.y;
      __nv_bfloat162 v = __floats2bfloat162_rn(o0, o1);
      o[e] = *reinterpret_cast<uint32_t*>(&v);
    }
    *reinterpret_cast<uint4*>(smg + (col >> 6) * TILE + r * 128 + ((((col & 63) >> 3) ^ (r & 7)) << 4)) = make_uint4(o[0], o[1], o[2], o[3]);
  }
  TSTAMP(11);
  fence_proxy_async();
  named_bar_sync(1 + wg, 128);
  if ((tid & 127) == 0) {
    for (int b = 0; b < GB; ++b) tma_store_2d(&mxa, smg + b * TILE, n0 + 64 * b, m0 + 64 * wg);
    tma_store_commit();
    tma_store_wait_read<0>();
  }
  TSTAMP(13);
  __syncwarp();
  cluster_wait();
  TSTAMP(6);
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
// 2-D map over a row-strided matrix [rows, cols] (cols contiguous), box [bo rows, bi cols], 128-B swizzle.
// Cached on (pointer, shape, stride, box): a descriptor is valid for one base pointer only.
const CUtensorMap& tile_map(const torch::Tensor& t, uint32_t bi, uint32_t bo) {
  static std::map<std::array<uint64_t, 6>, CUtensorMap> cache;
  static std::mutex lock;
  const std::array<uint64_t, 6> key{reinterpret_cast<uint64_t>(t.data_ptr()), (uint64_t)t.size(0), (uint64_t)t.size(1),
                                    (uint64_t)t.stride(0), bi, bo};
  std::lock_guard<std::mutex> g(lock);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  const bool f32 = t.scalar_type() == torch::kFloat32;
  CUtensorMap map{};
  const cuuint64_t dims[2] = {(cuuint64_t)t.size(1), (cuuint64_t)t.size(0)};
  const cuuint64_t strides[1] = {(cuuint64_t)t.stride(0) * t.element_size()};
  const cuuint32_t box[2] = {bi, bo}, elem[2] = {1, 1};
  TORCH_CHECK(encoder()(&map, f32 ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), dims,
                        strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS,
              "cuTensorMapEncodeTiled failed");
  return cache.emplace(key, map).first->second;
}

template <int NWG, bool ADALN>
void launch(const torch::Tensor& a, const torch::Tensor& w, torch::Tensor& x, const torch::Tensor& gl,
            const torch::Tensor* ms, const torch::Tensor* mb, torch::Tensor* xa, int64_t L, double eps) {
  using C = Cfg<NWG>;
  const int M = a.size(0), K = a.size(1);
  TORCH_CHECK((K / KC) % C::ST == 0, "K must be a whole number of ring turns: K % ", KC * C::ST, " != 0");
  TORCH_CHECK(L % C::BM == 0 && M % L == 0, "rows of a tile must not wrap the token axis: L % ", C::BM, " != 0");
  auto kern = gra_kernel<NWG, ADALN>;
  static bool attr = [&] { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM); return true; }();
  (void)attr;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(CL * (M / C::BM));
  cfg.blockDim = dim3(128 * (NWG + 1));
  cfg.dynamicSmemBytes = C::SMEM;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = CL; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  const torch::Tensor& xo = xa ? *xa : x;                          // unused map when !ADALN
  auto bp = [](const torch::Tensor* t) { return t ? reinterpret_cast<const __nv_bfloat16*>(t->data_ptr()) : nullptr; };
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, kern, tile_map(a, 64, C::BM / CL), tile_map(w, 64, BN), tile_map(x, 32, 64),
                                 tile_map(gl, 64, 64), xa ? tile_map(xo, 64, 64) : tile_map(x, 32, 64), bp(ms), bp(mb),
                                 (int)L, K, ms ? (int)ms->stride(0) : 0, mb ? (int)mb->stride(0) : 0,
                                 (float)eps) == cudaSuccess, "launch failed");
}
}  // namespace

// x += sigmoid(gl) * (a @ w^T); then, when ms is given, xa = AdaLN(x). nwg: consumer warpgroups (tile rows = 64 nwg).
void gemm_resgate_adaln(torch::Tensor a, torch::Tensor w, torch::Tensor x, torch::Tensor gl,
                        c10::optional<torch::Tensor> ms, c10::optional<torch::Tensor> mb, c10::optional<torch::Tensor> xa,
                        int64_t L, double eps, int64_t nwg) {
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16 && w.scalar_type() == torch::kBFloat16 && a.stride(1) == 1 && w.is_contiguous());
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 && x.is_contiguous() && x.size(1) == D_ && w.size(0) == D_);
  TORCH_CHECK(a.size(1) == w.size(1) && a.size(1) % KC == 0 && gl.stride(1) == 1 && gl.size(0) == L);
  const bool ad = ms.has_value();
  if (ad) TORCH_CHECK(xa->is_contiguous() && xa->scalar_type() == torch::kBFloat16 && ms->stride(1) == 1 && mb->stride(1) == 1);
  const torch::Tensor *pm = ad ? &*ms : nullptr, *pb = ad ? &*mb : nullptr;
  torch::Tensor* po = ad ? &*xa : nullptr;
  if (nwg == 1) { ad ? launch<1, true>(a, w, x, gl, pm, pb, po, L, eps) : launch<1, false>(a, w, x, gl, pm, pb, po, L, eps); }
  else { ad ? launch<2, true>(a, w, x, gl, pm, pb, po, L, eps) : launch<2, false>(a, w, x, gl, pm, pb, po, L, eps); }
}

torch::Tensor debug_times() {
  auto t = torch::zeros({1024, 16}, torch::kInt64);
#ifdef GRA_TIMES
  cudaMemcpyFromSymbol(t.data_ptr(), g_times, sizeof(g_times));
#endif
  return t;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_resgate_adaln", &gemm_resgate_adaln);
  m.def("debug_times", &debug_times);
}
