// gemm_resgate_adaln_ws.cu -- the fused residual GEMM with a warp-specialised epilogue (sm_90a). Same contract:
//
//   acc = A W^T                    A [M,K] bf16 (row stride sa), W [768,K] bf16
//   x  += sigmoid(gl[tok]) * acc   x [M,768] fp32, in place
//   xa  = LN(x) * sigmoid(ms[tok]) + mb[tok]        bf16, only when ADALN
//
// What _pp.cu / _pp3.cu measured: the mainloop is bounded by what an SM can ingest, so it wants pp's 64 x 192 tiles,
// and on the MMA warpgroups themselves a tile's epilogue (~11 us) is twice its mainloop, so it cannot hide. Here the
// epilogue has warpgroups of its own: the two MMA warpgroups ping-pong exactly as in pp and hand each finished tile
// to their paired epilogue warpgroup through shared memory (y, bf16 -- the same rounding the mm + rows path has), and
// go straight on to their next mainloop. The epilogue reads y from shared memory and x / gl / ms / mb / xa with
// 16-byte accesses along rows (8 lanes a row, 4 rows at a time), not in the wgmma fragment's scattered layout.
//
// Warpgroups: 0, 1 MMA; 2 epilogue (takes both MMA warpgroups' tiles, in order); 3: warp 0 the producer, warps 1-3
// three more epilogue warps (seven in all; the producer needs one thread). 512 threads, so ptxas's
// own budget is 128 registers: the MMA warpgroups' 96-register accumulator fits without setmaxnreg. (With a second
// epilogue warpgroup -- 640 threads, a 96-register budget -- the accumulator spilled 656 bytes of stack and the
// mainloop took 59 us a tile: ptxas did not budget the MMA region by setmaxnreg.inc.)
// 4-CTA cluster along N (192 columns each).
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;

#ifndef STAGES
#define STAGES 5
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

constexpr int D_ = 768, NC = 192, CL = 4, KC = 64, TM = 64, NTHR = 512;
constexpr int SA = TM * 128, SB = NC * 128, SS = SA + SB;           // ring stage: A 8 KB + W 24 KB
constexpr int YROW = NC * 2 + 16, YB = TM * YROW;                   // y tile rows padded: fragment writes conflict-free
constexpr int OY = STAGES * SS;
constexpr int OST = OY + 2 * YB;                                    // stats [tile parity][CL src][TM] (mean, M2)
constexpr int OBAR = OST + 2 * CL * TM * 8;
constexpr int SMEM_BYTES = OBAR + 256 + 1024;
constexpr uint32_t STAT_BYTES = CL * TM * 8;

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
__global__ void __launch_bounds__(NTHR, 1)
graws_kernel(const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mw,
             const __nv_bfloat16* __restrict__ GL, const __nv_bfloat16* __restrict__ MS,
             const __nv_bfloat16* __restrict__ MB, __nv_bfloat16* __restrict__ XA, float* __restrict__ X,
             int L, int K, int ntiles, int sgl, int sms, int smb, float eps) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + OBAR);
  uint64_t* empty = full + STAGES;
  uint64_t* yfull = empty + STAGES;                                 // [e]: MMA e's y tile is in ybuf[e]
  uint64_t* yfree = yfull + 2;                                      // [e]: the epilogue is done with that tile entirely
  uint64_t* sbar = yfree + 2;                                       // [tile parity]: the cluster's row stats landed
  uint64_t* turn = sbar + 4;                                        // [e]: the other MMA warpgroup is past its chunks
  float2* stats = reinterpret_cast<float2*>(sm + OST);

  const int tid = threadIdx.x, wg = tid >> 7;
  const uint32_t nr = cluster_rank();
  const int cid = cluster_id_x(), ncl = nclusters_x();
  const int n0 = nr * NC, nk = K / KC;
  const int nloc = cid < ntiles ? (ntiles - cid + ncl - 1) / ncl : 0;
  const int total = nloc * nk;

  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], CL * 4); }
    for (int e = 0; e < 2; ++e) { mbar_init(&yfull[e], 1); mbar_init(&yfree[e], 1); mbar_init(&turn[e], 1); }
    for (int i = 0; i < 2; ++i) mbar_init(&sbar[i], 1);
    fence_barrier_init();
  }
  __syncthreads();
  if (ADALN && tid == 0)
    for (int i = 0; i < 2; ++i) mbar_arrive_expect_tx(&sbar[i], STAT_BYTES);
  cluster_arrive(); cluster_wait();
#if PDL
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif

  const int lane = tid & 31, warp = (tid >> 5) & 3, ti = tid & 127;
  if (wg == 3 && warp == 0) {                                       // ---- producer: one thread
    if (tid == 384) {
      tma_prefetch_desc(&ma); tma_prefetch_desc(&mw);
      for (int i = 0; i < nloc; ++i) {
        const int m0 = (cid + i * ncl) * TM;
        for (int kc = 0; kc < nk; ++kc) {
          const int c = i * nk + kc, s = c % STAGES;
          mbar_wait(&empty[s], ((c / STAGES) & 1) ^ 1);
          mbar_arrive_expect_tx(&full[s], SS);
          tma_load_2d_mc(sm + s * SS + nr * (SA / CL), &ma, &full[s], kc * KC, m0 + nr * (TM / CL), (uint16_t)0xf);
          tma_load_2d(sm + s * SS + SA, &mw, &full[s], kc * KC, n0);
        }
      }
    }
    __syncwarp();
    cluster_arrive_relaxed(); cluster_wait();
    return;
  }

#ifdef PP_TIMES
  if (tid == 0) g_pp[blockIdx.x][15][0] = gtimer();
#endif
  if (wg < 2) {                                                     // ---- MMA warpgroup e = wg
    const int e = wg;
    const int rr0 = warp * 16 + (lane >> 2), cb = 2 * (lane & 3);
    const uint32_t sbase = smem_u32(sm);
    uint8_t* yb = sm + OY + e * YB;
    for (int i = e, j = 0; i < nloc; i += 2, ++j) {
      float acc[3][32];
      PST(i, 0);
      if (i > 0) mbar_wait(&turn[e], ((i - 1) >> 1) & 1);
      // Backpressure: start tile i only once the epilogue has finished tile i - 2 (this warpgroup's previous one).
      // That frees ybuf[e] for this tile's y, and it also bounds the peers: their stats for tile i can only arrive
      // after this CTA's MMA consumed tile i's chunks, i.e. after the epilogue is done reading tile i - 2's (same
      // parity). Gating only the y write let the MMA run ahead and peers overwrite stats still being read.
      if (j > 0) mbar_wait(&yfree[e], (j - 1) & 1);
      PST(i, 1);
      for (int kc = 0; kc < nk; ++kc) {
        const int c = i * nk + kc, s = c % STAGES;
        mbar_wait(&full[s], (c / STAGES) & 1);
        const uint32_t a0 = sbase + s * SS, b0 = a0 + SA;
        wgmma_fence();
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
#pragma unroll
          for (int q = 0; q < 3; ++q) mma64(acc[q], dsw(a0 + ks * 32), dsw(b0 + q * 8192 + ks * 32), (kc | ks) != 0);
        wgmma_commit();
        wgmma_wait<1>();
        if (kc > 0 && c - 1 + STAGES < total && lane == 0)
#pragma unroll
          for (int k = 0; k < CL; ++k) mbar_arrive_remote(&empty[(c - 1) % STAGES], k);
      }
      named_bar_sync(1 + e, 128);
      if (ti == 0 && i + 1 < nloc) mbar_arrive(&turn[e ^ 1]);
      PST(i, 2);
      wgmma_wait<0>();
      {
        const int c = i * nk + nk - 1;
        if (c + STAGES < total && lane == 0)
#pragma unroll
          for (int k = 0; k < CL; ++k) mbar_arrive_remote(&empty[c % STAGES], k);
      }
#pragma unroll
      for (int q = 0; q < 3; ++q) fence_regs(acc[q]);
      PST(i, 3);                                                    // hand the tile over: y (bf16) into ybuf[e]
#pragma unroll
      for (int q = 0; q < 3; ++q)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int g = 0; g < 8; ++g)
            *reinterpret_cast<__nv_bfloat162*>(yb + (rr0 + 8 * h) * YROW + (q * 64 + g * 8 + cb) * 2) =
                __floats2bfloat162_rn(acc[q][4 * g + 2 * h], acc[q][4 * g + 2 * h + 1]);
      named_bar_sync(1 + e, 128);
      if (ti == 0) mbar_arrive(&yfull[e]);
      PST(i, 4);
    }
  } else {                                                          // ---- epilogue: 7 warps, every tile, in order
    // A tile is 16 groups of 4 rows; epilogue warp ew takes groups ew, ew + 7, ew + 14. 8 lanes a row: chunks cl,
    // cl + 8, cl + 16 of 8 columns.
    constexpr int NEW = 7, NGRP = TM / 4, NPASS = (NGRP + NEW - 1) / NEW;
    const int ew = wg == 2 ? warp : 3 + warp;
    const bool lead = ew == 0 && lane == 0;
    const int rl = lane >> 3, cl = lane & 7;
    for (int i = 0; i < nloc; ++i) {
      const int m0 = (cid + i * ncl) * TM, t0 = m0 % L;
      const int e = i & 1, j = i >> 1;                              // from MMA e, its j-th tile
      const uint8_t* yb = sm + OY + e * YB;
      float2* st = stats + e * CL * TM;
      uint64_t* sb = &sbar[e];
      mbar_wait(&yfull[e], j & 1);
      PST(i, 5);
      // Both passes are double-buffered over the four row groups: the next group's global loads are in flight while
      // this one computes (one group at a time left ~1.2 us of latency exposed per group: ~10 us a tile, twice the
      // mainloop, with a single epilogue warpgroup serving both MMA warpgroups).
      float4 xb[2][3][2]; uint4 gb[2][3];
      auto ld_res = [&](int p, int bf) {
        const int r = (ew + NEW * p) * 4 + rl;
        const float* xr = X + (size_t)(m0 + r) * D_ + n0;
        const __nv_bfloat16* gr = GL + (size_t)(t0 + r) * sgl + n0;
#pragma unroll
        for (int k = 0; k < 3; ++k) {
          const int col = 8 * (cl + 8 * k);
          xb[bf][k][0] = *reinterpret_cast<const float4*>(xr + col);
          xb[bf][k][1] = *reinterpret_cast<const float4*>(xr + col + 4);
          gb[bf][k] = __ldg(reinterpret_cast<const uint4*>(gr + col));
        }
      };
      ld_res(0, 0);
#pragma unroll
      for (int p = 0; p < NPASS; ++p) {                             // residual, 4 rows of this warp at a time
        if (ew + NEW * p >= NGRP) break;                            // warp-uniform
        const int bf = p & 1;
        if (p + 1 < NPASS && ew + NEW * (p + 1) < NGRP) ld_res(p + 1, bf ^ 1);
        const int r = (ew + NEW * p) * 4 + rl;
        float* xr = X + (size_t)(m0 + r) * D_ + n0;
        float v[3][8];
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < 3; ++k) {
          const int col = 8 * (cl + 8 * k);
          const uint4 yv = *reinterpret_cast<const uint4*>(yb + r * YROW + col * 2);
          const float4 x0 = xb[bf][k][0], x1 = xb[bf][k][1];
          const float xs[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
          const uint32_t yw[4] = {yv.x, yv.y, yv.z, yv.w}, gw[4] = {gb[bf][k].x, gb[bf][k].y, gb[bf][k].z, gb[bf][k].w};
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            const float2 yy = bf2f(yw[u]), gg = bf2f(gw[u]);
            v[k][2 * u] = xs[2 * u] + sigmoid_t(gg.x) * yy.x;
            v[k][2 * u + 1] = xs[2 * u + 1] + sigmoid_t(gg.y) * yy.y;
            s += v[k][2 * u] + v[k][2 * u + 1];
          }
          *reinterpret_cast<float4*>(xr + col) = make_float4(v[k][0], v[k][1], v[k][2], v[k][3]);
          *reinterpret_cast<float4*>(xr + col + 4) = make_float4(v[k][4], v[k][5], v[k][6], v[k][7]);
        }
        if (ADALN) {
          s += __shfl_xor_sync(0xffffffffu, s, 1);
          s += __shfl_xor_sync(0xffffffffu, s, 2);
          s += __shfl_xor_sync(0xffffffffu, s, 4);
          const float mean = s * (1.f / NC);
          float m2 = 0.f;
#pragma unroll
          for (int k = 0; k < 3; ++k)
#pragma unroll
            for (int u = 0; u < 8; ++u) { const float d = v[k][u] - mean; m2 += d * d; }
          m2 += __shfl_xor_sync(0xffffffffu, m2, 1);
          m2 += __shfl_xor_sync(0xffffffffu, m2, 2);
          m2 += __shfl_xor_sync(0xffffffffu, m2, 4);
          if (cl == 0)
#pragma unroll
            for (int k = 0; k < CL; ++k) st_async_f2(&st[nr * TM + r], sb, k, mean, m2);
        }
      }
      PST(i, 6);
      if (ADALN) {
        uint4 sbv[2][3], bbv[2][3];
        auto ld_ada = [&](int p, int bf) {                          // x again (this thread's own stores), ms, mb
          const int r = (ew + NEW * p) * 4 + rl;
          const float* xr = X + (size_t)(m0 + r) * D_ + n0;
          const __nv_bfloat16* sr = MS + (size_t)(t0 + r) * sms + n0;
          const __nv_bfloat16* br = MB + (size_t)(t0 + r) * smb + n0;
#pragma unroll
          for (int k = 0; k < 3; ++k) {
            const int col = 8 * (cl + 8 * k);
            xb[bf][k][0] = *reinterpret_cast<const float4*>(xr + col);
            xb[bf][k][1] = *reinterpret_cast<const float4*>(xr + col + 4);
            sbv[bf][k] = __ldg(reinterpret_cast<const uint4*>(sr + col));
            bbv[bf][k] = __ldg(reinterpret_cast<const uint4*>(br + col));
          }
        };
        ld_ada(0, 0);                                               // does not need the stats: under their wait
        mbar_wait(sb, j & 1);
        PST(i, 7);
#pragma unroll
        for (int p = 0; p < NPASS; ++p) {
          if (ew + NEW * p >= NGRP) break;
          const int bf = p & 1;
          if (p + 1 < NPASS && ew + NEW * (p + 1) < NGRP) ld_ada(p + 1, bf ^ 1);
          const int r = (ew + NEW * p) * 4 + rl;
          float2 pp[CL];
#pragma unroll
          for (int k = 0; k < CL; ++k) pp[k] = st[k * TM + r];
          float mu = 0.f;
#pragma unroll
          for (int k = 0; k < CL; ++k) mu += pp[k].x;
          mu *= 1.f / CL;
          float M2 = 0.f;
#pragma unroll
          for (int k = 0; k < CL; ++k) { const float dm = pp[k].x - mu; M2 += pp[k].y + float(NC) * dm * dm; }
          const float rstd = rsqrtf(M2 * (1.f / D_) + eps);
          __nv_bfloat16* ar = XA + (size_t)(m0 + r) * D_ + n0;
#pragma unroll
          for (int k = 0; k < 3; ++k) {
            const int col = 8 * (cl + 8 * k);
            const float4 x0 = xb[bf][k][0], x1 = xb[bf][k][1];
            const float xs[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
            const uint32_t sw[4] = {sbv[bf][k].x, sbv[bf][k].y, sbv[bf][k].z, sbv[bf][k].w};
            const uint32_t bw[4] = {bbv[bf][k].x, bbv[bf][k].y, bbv[bf][k].z, bbv[bf][k].w};
            uint32_t o[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
              const float2 sc = bf2f(sw[u]), sh = bf2f(bw[u]);
              __nv_bfloat162 t = __floats2bfloat162_rn((xs[2 * u] - mu) * rstd * sigmoid_t(sc.x) + sh.x,
                                                       (xs[2 * u + 1] - mu) * rstd * sigmoid_t(sc.y) + sh.y);
              o[u] = *reinterpret_cast<uint32_t*>(&t);
            }
            *reinterpret_cast<uint4*>(ar + col) = make_uint4(o[0], o[1], o[2], o[3]);
          }
        }
      }
      named_bar_sync(3, 32 * NEW);                                  // every thread is past y and past the stats
      if (lead) {
        if (ADALN) mbar_arrive_expect_tx(sb, STAT_BYTES);         // re-armed for this parity's next tile
        mbar_arrive(&yfree[e]);                                     // MMA e may start its next tile
      }
      PST(i, 8);
    }
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
  auto kern = graws_kernel<ADALN>;
  static int maxc = [&] {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    cudaLaunchConfig_t q{};
    q.gridDim = dim3(CL * 32); q.blockDim = dim3(NTHR); q.dynamicSmemBytes = SMEM_BYTES;  // NTHR = 512
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

void gemm_resgate_adaln_ws(torch::Tensor a, torch::Tensor w, torch::Tensor x, torch::Tensor gl,
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
  m.def("gemm_resgate_adaln_ws", &gemm_resgate_adaln_ws);
  m.def("debug_times", &debug_times);
}
