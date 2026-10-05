// attn_dqb.cu -- the backward's dQ pass WITH dbias (K2'), bf16 operands, fp32 accumulation.
//
// dbias = sum over the A samples of dS. attn_dkv.cu adds every sample's dS into L2 with atomics (~680 us at L768/A48:
// L2 atomics, not bytes, are the cost). Here the CTA's G = 3 consumer warpgroups are G SAMPLES of the same 64 queries
// and head: each streams its own sample's K and V against the one bias tile, and after every key block the three dS
// tiles meet in shared memory, where one warpgroup (rotating with the block) sums them and issues the L2 reds -- a
// third of the atomics, and no cluster: the exchange is two named barriers inside the CTA.
//
// Per key block, warpgroup w (sample a0 + w):  S = q K^T + bias - LSE (log2), P = 2^S, dP = dO V^T, dS = P (dP - D),
// dQ += dS K / sqrt 48, and dS (fp32) -> the exchange.
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;

#ifndef STAGES
#define STAGES 3
#endif
#ifndef QWID
#define QWID 48
#endif
#ifndef DBR
#define DBR 1                    // timing diagnostics: 0 skips the dbias L2 reds (keeps the exchange)
#endif
#ifndef DBX
#define DBX 1                    // timing diagnostics: 0 skips the whole dbias exchange and reds
#endif
#ifndef DBTMA
#define DBTMA 2                  // dbias, L768 A48: 0 = the G-sample exchange + L2 reds by the owner (1019 us); 1 = every
#endif                           // warpgroup TMA-reduce-adds its own tile, no exchange (1185: 3x the reduce traffic); 2 = the
                                 // exchange, then the owner stages the summed tile and ONE TMA reduce-add replaces its reds
#ifndef SPLITRED
#define SPLITRED 0               // 1: every warpgroup reduces a third of the dS tile; 0: one owner warpgroup a block
#endif
constexpr int G = 3, BN = 64, QB = 64, DH = 48, QW = QWID, DM = 768;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float RSQD = 0.14433756729740643f;                         // 1 / sqrt(48)

TMN_DEVI uint64_t dsw(uint32_t addr) { return smem_desc(addr, 16, 1024, 1); }          // K-major, 128-B swizzle
TMN_DEVI uint64_t dmn(uint32_t base, int ks) { return smem_desc(base + ks * 2048, 16, 1024, 1); }   // MN-major, k-step = 16 rows
TMN_DEVI uint32_t sw128(int row, int byte) { return row * 128 + ((((byte >> 4) ^ (row & 7))) << 4) + (byte & 15); }
TMN_DEVI float ex2(float a) { float r; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }

TMN_DEVI void mma_s_rs(float* d, const uint32_t (&a)[4], uint64_t b, int accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %37, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 0; }"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(accumulate));
}
TMN_DEVI void mma_o(float (&d)[24], const uint32_t (&a)[4], uint64_t b) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23}, {%24,%25,%26,%27}, %28, p, 1, 1, 1; }"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b));
}
TMN_DEVI void tma_load_mc(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1, uint16_t mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;"
               :: "r"(smem_u32(dst)), "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "h"(mask) : "memory");
}
TMN_DEVI uint32_t mapa(uint32_t a, uint32_t rank) {
  uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank)); return r;
}
TMN_DEVI void mbar_arrive_remote(uint64_t* bar, uint32_t rank) {
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(mapa(smem_u32(bar), rank)) : "memory");
}
TMN_DEVI void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
TMN_DEVI uint32_t cluster_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r)); return r; }
TMN_DEVI float2 bf2f(uint32_t u) { return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u)); }
TMN_DEVI float qmax(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}
TMN_DEVI float qsum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

TMN_DEVI void bulk_reduce_add_2d(const CUtensorMap* map, uint32_t src, int c0, int c1) {
  asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%2, %3}], [%1];"
               :: "l"(map), "r"(src), "r"(c0), "r"(c1) : "memory");
}
TMN_DEVI void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
TMN_DEVI void bulk_wait_read0() { asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory"); }

TMN_DEVI void red_v2(float* p, float a, float b) {
  if (!DBR) { if (a == 1234.5f && b == -1.f) *p = a; return; }  // keeps the sum alive
  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(p), "f"(a), "f"(b) : "memory");
}

// smem: STAGES slots of [K_0 K_1 K_2 V_0 V_1 V_2 bias] (8 KB tiles, 64 rows x 128 B), the dS exchange (G x 16 KB), bars.
// The G q and dO tiles are parked in slot 0 before the key loop.
constexpr int TILE = 64 * 128, SLOT = (2 * G + 1) * TILE, OFF_B = 2 * G * TILE;
constexpr int OFF_X = STAGES * SLOT, SX = 128 * 32 * 4, OFF_BAR = OFF_X + G * SX;
static_assert(2 * G * TILE <= SLOT, "q / dO parking");

template <bool HASM>
__global__ void __launch_bounds__(128 * G + 32, 1)
attn_dqb_kernel(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mk,
                const __grid_constant__ CUtensorMap mv, const __grid_constant__ CUtensorMap mdo,
                const __grid_constant__ CUtensorMap mbias, const float* __restrict__ KM, const float* __restrict__ LSE,
                const float* __restrict__ DD, float* __restrict__ DQ, float* __restrict__ DB, const __grid_constant__ CUtensorMap mdb,
                int L, int H, int mt) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + OFF_BAR);
  uint64_t* empty = full + STAGES;
  uint64_t* qbar = empty + STAGES;
  uint64_t* qdone = qbar + 1;

  const int tid = threadIdx.x;
  int bid = blockIdx.x;
  const int m_tile = bid % mt; bid /= mt;
  const int head = bid % H, grp = bid / H;
  const int a0 = grp * G, m0 = m_tile * QB, qcol = head * DH;
  const int wg = tid >> 7;
  const int nblocks = L / BN;

  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], 4 * G); }
    mbar_init(qbar, 1);
    mbar_init(qdone, 4 * G);
    fence_barrier_init();
  }
  __syncthreads();

  if (tid >= 128 * G) {                                             // producer: one thread
    if (tid == 128 * G) {
      tma_prefetch_desc(&mq); tma_prefetch_desc(&mk); tma_prefetch_desc(&mv); tma_prefetch_desc(&mdo);
      tma_prefetch_desc(&mbias);
      mbar_arrive_expect_tx(qbar, 2 * G * QB * QW * 2);
      for (int w = 0; w < G; ++w) {
        tma_load_2d(sm + w * TILE, &mq, qbar, qcol, (a0 + w) * L + m0);
        tma_load_2d(sm + (G + w) * TILE, &mdo, qbar, qcol, (a0 + w) * L + m0);
      }
      mbar_wait(qdone, 0);
      for (int n = 0; n < nblocks; ++n) {
        const int s = n % STAGES;
        mbar_wait(&empty[s], ((n / STAGES) & 1) ^ 1);
        uint8_t* slot = sm + s * SLOT;
        mbar_arrive_expect_tx(&full[s], 2 * G * BN * QW * 2 + TILE);
        for (int w = 0; w < G; ++w) {
          tma_load_2d(slot + w * TILE, &mk, &full[s], qcol, (a0 + w) * L + n * BN);
          tma_load_2d(slot + (G + w) * TILE, &mv, &full[s], qcol, (a0 + w) * L + n * BN);
        }
        tma_load_2d(slot + OFF_B, &mbias, &full[s], n * BN, head * L + m0);
      }
    }
    return;
  }

  const int lane = tid & 31, warp = (tid >> 5) & 3, wt = tid & 127;
  const int r0 = warp * 16 + (lane >> 2), cb = 2 * (lane & 3);
  const int a = a0 + wg;
  float acc[24];
#pragma unroll
  for (int i = 0; i < 24; ++i) acc[i] = 0.f;
  float lse[2], dd[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const size_t ri = ((size_t)a * H + head) * L + m0 + r0 + 8 * h;
    lse[h] = LSE[ri]; dd[h] = DD[ri];
  }
  const uint32_t sbase = smem_u32(sm);
  mbar_wait(qbar, 0);
  uint32_t qr[DH / 16][4], dor[DH / 16][4];
  {
    uint32_t dep = 0;
#pragma unroll
    for (int ks = 0; ks < DH / 16; ++ks) {
      const int rr = warp * 16 + 8 * ((lane >> 3) & 1) + (lane & 7), bb = (16 * ks + 8 * (lane >> 4)) * 2;
      ldsm_x4(qr[ks], sbase + wg * TILE + sw128(rr, bb));
      ldsm_x4(dor[ks], sbase + (G + wg) * TILE + sw128(rr, bb));
      dep ^= qr[ks][0] ^ qr[ks][1] ^ qr[ks][2] ^ qr[ks][3] ^ dor[ks][0] ^ dor[ks][1] ^ dor[ks][2] ^ dor[ks][3];
    }
    fence_proxy_async();                                            // the parked q / dO read generically: order before release
    __syncwarp();
    if (lane == 0) mbar_arrive_dep(qdone, zero_dep(dep));
  }
  const float* kmr = HASM ? KM + (size_t)a * L : nullptr;
  const uint64_t dK0 = dsw(sbase), dM0 = dmn(sbase, 0);
  float* dbrow = DB + ((size_t)head * L + m0 + r0) * L;             // dbias [H, L(query), L(key)], natural layout
  float4* xs = reinterpret_cast<float4*>(sm + OFF_X);

  float sc[32], dp[32];
  uint32_t pa[BN / 16][4];
  for (int n = 0; n < nblocks; ++n) {
    const int s = n % STAGES;
    mbar_wait(&full[s], (n / STAGES) & 1);
    {
      const uint32_t sbn = sbase + s * SLOT + OFF_B;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint32_t bm[2][4];
#pragma unroll
        for (int half = 0; half < 2; ++half)
          ldsm_x4(bm[half], sbn + sw128(16 * warp + 8 * h + (lane & 7), 8 * (4 * half + (lane >> 3)) * 2));
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float2 b = bf2f(bm[j >> 2][j & 3]);
          float m0v = -lse[h], m1v = -lse[h];
          if (HASM) {
            const float2 t = *reinterpret_cast<const float2*>(kmr + n * BN + j * 8 + cb);
            m0v += t.x; m1v += t.y;
          }
          sc[4 * j + 2 * h] = b.x + m0v;
          sc[4 * j + 2 * h + 1] = b.y + m1v;
        }
      }
    }
    wgmma_fence();
#pragma unroll
    for (int ks = 0; ks < DH / 16; ++ks) mma_s_rs(sc, qr[ks], dK0 + ((s * SLOT + wg * TILE + ks * 32) >> 4), 1);
#pragma unroll
    for (int ks = 0; ks < DH / 16; ++ks) mma_s_rs(dp, dor[ks], dK0 + ((s * SLOT + (G + wg) * TILE + ks * 32) >> 4), ks != 0);
    wgmma_commit();
    wgmma_wait<0>();
#pragma unroll
    for (int i = 0; i < 32; ++i) { fence_reg(sc[i]); fence_reg(dp[i]); }
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float p0 = ex2(sc[4 * j + 2 * h]), p1 = ex2(sc[4 * j + 2 * h + 1]);
        const float s0 = p0 * (dp[4 * j + 2 * h] - dd[h]), s1 = p1 * (dp[4 * j + 2 * h + 1] - dd[h]);
        sc[4 * j + 2 * h] = s0; sc[4 * j + 2 * h + 1] = s1;         // sc now holds dS (fp32)
        const __nv_bfloat162 pk = __floats2bfloat162_rn(s0, s1);
        pa[j >> 1][2 * (j & 1) + h] = *reinterpret_cast<const uint32_t*>(&pk);
      }
    wgmma_fence();
#pragma unroll
    for (int i = 0; i < 24; ++i) fence_reg(acc[i]);
#pragma unroll
    for (int ks = 0; ks < BN / 16; ++ks) mma_o(acc, pa[ks], dM0 + ((s * SLOT + wg * TILE + ks * 2048) >> 4));
    wgmma_commit();
    // ---- dbias: the three samples' dS meet in shared memory; warpgroup n % G sums them and issues the reds
    if (DBX && DBTMA == 1) {
      // this sample's dS tile -> a 128-B-swizzled staging tile (two 32-column halves) -> TMA reduce-add into DB
      uint8_t* xt = sm + OFF_X + wg * SX;
      if (n > 0) {
        if (wt == 0) bulk_wait_read0();                             // the previous tile has left the staging buffer
        named_bar_sync(3 + wg, 128);
      }
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int j = 0; j < 8; ++j)
          *reinterpret_cast<float2*>(xt + (j >> 2) * 8192 + sw128(r0 + 8 * h, ((j & 3) * 8 + cb) * 4)) =
              make_float2(sc[4 * j + 2 * h], sc[4 * j + 2 * h + 1]);
      fence_proxy_async();
      named_bar_sync(3 + wg, 128);
      if (wt == 0) {
        bulk_reduce_add_2d(&mdb, smem_u32(xt), n * BN, head * L + m0);
        bulk_reduce_add_2d(&mdb, smem_u32(xt + 8192), n * BN + 32, head * L + m0);
        bulk_commit();
      }
    } else if (DBX) {
    if (DBTMA == 2 && n > 0 && (n - 1) % G == wg) {                  // the previous owner: its slot is the TMA source
      if (wt == 0) bulk_wait_read0();
      named_bar_sync(3 + wg, 128);
    }
    if (n > 0) named_bar_sync(2, 128 * G);                          // the previous block's owner has read the exchange
#pragma unroll
    for (int i = 0; i < 8; ++i) xs[(wg * 8 + i) * 128 + wt] = make_float4(sc[4 * i], sc[4 * i + 1], sc[4 * i + 2], sc[4 * i + 3]);
    named_bar_sync(1, 128 * G);
#if SPLITRED
    // every warpgroup sums and reds a third of the tile: the column groups j with j % G == (wg + n) % G (rotating, so
    // the two-group share moves around); float4 i of the exchange holds column group i (both row halves)
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (j % G != (wg + n) % G) continue;
      float4 t = make_float4(sc[4 * j], sc[4 * j + 1], sc[4 * j + 2], sc[4 * j + 3]);
#pragma unroll
      for (int o = 1; o < G; ++o) {
        const float4 f = xs[(((wg + o) % G) * 8 + j) * 128 + wt];
        t.x += f.x; t.y += f.y; t.z += f.z; t.w += f.w;
      }
      red_v2(dbrow + n * BN + j * 8 + cb, t.x, t.y);
      red_v2(dbrow + (size_t)8 * L + n * BN + j * 8 + cb, t.z, t.w);
    }
#else
    if (n % G == wg) {
#pragma unroll
      for (int o = 1; o < G; ++o) {
        const int w2 = (wg + o) % G;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float4 f = xs[(w2 * 8 + i) * 128 + wt];
          sc[4 * i] += f.x; sc[4 * i + 1] += f.y; sc[4 * i + 2] += f.z; sc[4 * i + 3] += f.w;
        }
      }
      if (DBTMA == 2) {                                             // own slot (nobody reads it) -> swizzled tile -> TMA reduce
        uint8_t* xt = sm + OFF_X + wg * SX;                         // (bar 1 above: every exchange write is done)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int j = 0; j < 8; ++j)
            *reinterpret_cast<float2*>(xt + (j >> 2) * 8192 + sw128(r0 + 8 * h, ((j & 3) * 8 + cb) * 4)) =
                make_float2(sc[4 * j + 2 * h], sc[4 * j + 2 * h + 1]);
        fence_proxy_async();
        named_bar_sync(3 + wg, 128);
        if (wt == 0) {
          bulk_reduce_add_2d(&mdb, smem_u32(xt), n * BN, head * L + m0);
          bulk_reduce_add_2d(&mdb, smem_u32(xt + 8192), n * BN + 32, head * L + m0);
          bulk_commit();
        }
      } else
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int j = 0; j < 8; ++j)
          red_v2(dbrow + (size_t)(8 * h) * L + n * BN + j * 8 + cb, sc[4 * j + 2 * h], sc[4 * j + 2 * h + 1]);
    }
#endif
    }
    wgmma_wait<0>();
    fence_proxy_async();                                          // generic (ldmatrix) reads of this TMA stage before its release
    __syncwarp();
    if (lane == 0 && n + STAGES < nblocks) mbar_arrive(&empty[s]);
  }
  if (DBTMA && wt == 0) bulk_wait_read0();                          // the last tile has been read before the CTA exits
#pragma unroll
  for (int i = 0; i < 24; ++i) fence_reg(acc[i]);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    float* orow = DQ + ((size_t)a * L + m0 + r0 + 8 * h) * DM + qcol;
#pragma unroll
    for (int j = 0; j < 6; ++j)
      *reinterpret_cast<float2*>(orow + j * 8 + cb) = make_float2(acc[4 * j + 2 * h] * RSQD, acc[4 * j + 2 * h + 1] * RSQD);
  }
}

// ------------------------------------------------------------------------------------------------ host
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

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
CUtensorMap tile_map(void* ptr, uint64_t rows, uint64_t cols, uint64_t stride, uint32_t bi, uint32_t bo) {
  CUtensorMap map{};
  const cuuint64_t dims[2] = {cols, rows};
  const cuuint64_t strides[1] = {stride * 2};
  const cuuint32_t box[2] = {bi, bo}, elem[2] = {1, 1};
  TORCH_CHECK(encoder()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, dims, strides, box, elem,
                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS, "encode failed");
  return map;
}
}  // namespace

// as attn_dq.cu, plus DB [H, L, L] fp32 (zeroed by the caller): dbias ADDED in natural units, natural layout
void attn_dqb(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dO, torch::Tensor bias,
              c10::optional<torch::Tensor> kmask, torch::Tensor LSE, torch::Tensor Dd, torch::Tensor DQ, torch::Tensor DB) {
  const int H = bias.size(0), L = bias.size(1);
  const int A = q.size(0) / L;
  for (auto* t : {&q, &k, &v, &dO})
    TORCH_CHECK(t->is_contiguous() && t->scalar_type() == torch::kBFloat16 && t->size(1) == DM && t->size(0) == A * L, "qkv layout");
  TORCH_CHECK(bias.is_contiguous() && bias.scalar_type() == torch::kBFloat16 && bias.size(2) == L && H * DH == DM, "bias layout");
  for (auto* t : {&DQ, &DB}) TORCH_CHECK(t->is_contiguous() && t->scalar_type() == torch::kFloat32, "out layout");
  TORCH_CHECK(L % QB == 0 && L % BN == 0 && A % G == 0, "shape");
  const size_t smem = 1024 + OFF_BAR + 256;
  const bool hasm = kmask.has_value() && kmask->defined();
  auto kern = hasm ? attn_dqb_kernel<true> : attn_dqb_kernel<false>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  const int mt = L / QB;
  auto mq = tile_map(q.data_ptr(), (uint64_t)A * L, DM, DM, QW, QB);
  auto mk = tile_map(k.data_ptr(), (uint64_t)A * L, DM, DM, QW, BN);
  auto mv = tile_map(v.data_ptr(), (uint64_t)A * L, DM, DM, QW, BN);
  auto mdo = tile_map(dO.data_ptr(), (uint64_t)A * L, DM, DM, QW, QB);
  auto mb = tile_map(bias.data_ptr(), (uint64_t)H * L, L, L, BN, QB);
  CUtensorMap mdb{};                                                // DB [H L, L] fp32, 32-column 128-B-swizzled boxes
  {
    const cuuint64_t dims[2] = {(cuuint64_t)L, (cuuint64_t)H * L}, strides[1] = {(cuuint64_t)L * 4};
    const cuuint32_t box[2] = {32, QB}, elem[2] = {1, 1};
    TORCH_CHECK(encoder()(&mdb, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, DB.data_ptr(), dims, strides, box, elem,
                          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS, "encode failed");
  }
  kern<<<(A / G) * H * mt, 128 * G + 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      mq, mk, mv, mdo, mb, hasm ? kmask->data_ptr<float>() : nullptr, LSE.data_ptr<float>(), Dd.data_ptr<float>(),
      DQ.data_ptr<float>(), DB.data_ptr<float>(), mdb, L, H, mt);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("attn_dqb", &attn_dqb); }
