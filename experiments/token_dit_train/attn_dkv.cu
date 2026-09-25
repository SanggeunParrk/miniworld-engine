// attn_dkv.cu -- the backward's dK / dV / dbias pass (K1), bf16 operands with fp32 accumulation.
//
// A CTA per (sample, head, 64 NWG keys); warpgroup w owns 64 keys (the M dimension) and streams every query block:
//   S^T  = K Q^T + (bias - LSE)^T   (log2 units, K and V held in registers as the A operands)
//   dP^T = V dO^T                   P^T = 2^S^T      dS^T = P^T (dP^T - D)
//   dV += P^T dO      dK += dS^T Q / log2 e         dbias += dS (summed over the samples with L2 atomics)
// dbias is accumulated TRANSPOSED, [H, L(key), L(query)], so a thread's two adjacent queries are one v2 red.
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;

#ifndef STAGES
#define STAGES 4
#endif
#ifndef NWG
#define NWG 2                    // consumer warpgroups: query rows per CTA = 64 NWG
#endif
#ifndef BLKSM
#define BLKSM 1
#endif
#ifndef NISS
#define NISS 3
#endif
#ifndef RSPLIT
#define RSPLIT 0                 // setmaxnreg split. Off: ptxas ignores it at this kernel's 96 registers anyway, and when
#endif                           // the kernel is allocated fewer registers than the launch bound (the FLOOR build) the
                                 // .inc asks for more than the CTA owns and waits for them forever
#ifndef FCLS
#define FCLS 1                    // samples per cluster: the bias tile is multicast to the FCLS CTAs that share it
#endif
#ifndef NOBIAS
#define NOBIAS 0                 // FLOOR diagnostics only: skip the bias tile (K / V traffic alone)
#endif
#ifndef LAZY
#define LAZY 8.0f                // running-max slack in log2 units (0: exact running max)
#endif
#ifndef QROT
#define QROT 1                   // start each CTA's query loop at a sample-dependent block, so the CTAs of different
#endif                           // samples do not hit the same dbias addresses at the same time
#ifndef RV4
#define RV4 1                    // 1: v4 reds (a quad shuffle gives each thread 4 adjacent queries), 0: v2
#endif
#ifndef DBIAS
#define DBIAS 1                  // 0: skip the dbias atomics (to price them)
#endif
#ifndef FLOOR
#define FLOOR 0                  // 1: the same TMA traffic and barriers, no math -- the measured pattern floor
#endif
constexpr int BN = 64, DH = 48, QW = 64, QM = 64 * NWG, DM = 768;
constexpr float LOG2E = 1.4426950408889634f;

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

TMN_DEVI void red_v2(float* p, float a, float b) {
  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(p), "f"(a), "f"(b) : "memory");
}
TMN_DEVI void red_v4(float* p, float a, float b, float c, float d) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}

// stage: Q (64 queries, 64 wide), dO (same), the bias tile of each warpgroup (64 queries x 64 keys), LSE and D (64 each)
constexpr int QB = 64, SQT = QB * 128, SBT = QB * 128;
constexpr int OFF_QT = 0, OFF_DOT = SQT, OFF_BT = 2 * SQT, OFF_LD = OFF_BT + NWG * SBT;
constexpr int ST_BYTES = (OFF_LD + 2 * QB * 4 + 1023) / 1024 * 1024;   // 128-B swizzled tiles need 1 KB slots
constexpr int OFF_KV = STAGES * ST_BYTES;                            // K and V tiles (64 NWG keys each), read once
constexpr int SKV = 64 * NWG * 128;

template <bool HASM>
__global__ void __launch_bounds__(128 * NWG + 32, BLKSM)
attn_dkv_kernel(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mk,
                const __grid_constant__ CUtensorMap mv, const __grid_constant__ CUtensorMap mdo,
                const __grid_constant__ CUtensorMap mbias, const float* __restrict__ KM, const float* __restrict__ LSE,
                const float* __restrict__ DD, float* __restrict__ DK, float* __restrict__ DV, float* __restrict__ DBT,
                int L, int H, int A, int nt) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + OFF_KV + 2 * SKV);
  uint64_t* empty = full + STAGES;
  uint64_t* kvbar = empty + STAGES;

  const int tid = threadIdx.x;
  const int samp = blockIdx.x % A, cid = blockIdx.x / A;
  const int k_tile = cid % nt, head = cid / nt;
  const int k0 = k_tile * 64 * NWG, qcol = head * DH;
  const int wg = tid >> 7;
  const int nblocks = L / QB;

  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], 4 * NWG); }
    mbar_init(kvbar, 1);
    fence_barrier_init();
  }
  __syncthreads();

  if (tid >= 128 * NWG) {                                           // producer: one thread issues everything
    if (tid == 128 * NWG) {
      tma_prefetch_desc(&mq); tma_prefetch_desc(&mk); tma_prefetch_desc(&mv); tma_prefetch_desc(&mdo);
      tma_prefetch_desc(&mbias);
      mbar_arrive_expect_tx(kvbar, 2 * SKV);
      tma_load_2d(sm + OFF_KV, &mk, kvbar, qcol, samp * L + k0);
      tma_load_2d(sm + OFF_KV + SKV, &mv, kvbar, qcol, samp * L + k0);
      for (int n = 0; n < nblocks; ++n) {
        const int s = n % STAGES;
        const int nb = QROT ? (n + samp) % nblocks : n;              // the query block this stage carries
        mbar_wait(&empty[s], ((n / STAGES) & 1) ^ 1);
        uint8_t* slot = sm + s * ST_BYTES;
        mbar_arrive_expect_tx(&full[s], OFF_LD + 2 * QB * 4);
        tma_load_2d(slot + OFF_QT, &mq, &full[s], qcol, samp * L + nb * QB);
        tma_load_2d(slot + OFF_DOT, &mdo, &full[s], qcol, samp * L + nb * QB);
        for (int w = 0; w < NWG; ++w)
          tma_load_2d(slot + OFF_BT + w * SBT, &mbias, &full[s], k0 + 64 * w, head * L + nb * QB);
        const size_t li = ((size_t)samp * H + head) * L + nb * QB;
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     :: "r"(smem_u32(slot + OFF_LD)), "l"(LSE + li), "r"(QB * 4), "r"(smem_u32(&full[s])) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     :: "r"(smem_u32(slot + OFF_LD + QB * 4)), "l"(DD + li), "r"(QB * 4), "r"(smem_u32(&full[s])) : "memory");
      }
    }
    return;
  }

  const int lane = tid & 31, warp = (tid >> 5) & 3;
  const int r0 = warp * 16 + (lane >> 2), cb = 2 * (lane & 3);
  float dk[24], dv[24];
#pragma unroll
  for (int i = 0; i < 24; ++i) { dk[i] = 0.f; dv[i] = 0.f; }
  mbar_wait(kvbar, 0);
  uint32_t kr[DH / 16][4], vr[DH / 16][4];
  {
    const uint32_t sk = smem_u32(sm) + OFF_KV + wg * 8192, sv = sk + SKV;
#pragma unroll
    for (int ks = 0; ks < DH / 16; ++ks) {
      const int rr = warp * 16 + 8 * ((lane >> 3) & 1) + (lane & 7), bb = (16 * ks + 8 * (lane >> 4)) * 2;
      ldsm_x4(kr[ks], sk + sw128(rr, bb));
      ldsm_x4(vr[ks], sv + sw128(rr, bb));
    }
  }
  float kmv[2] = {0.f, 0.f};                                         // the key mask of this thread's two key rows
  if (HASM) { kmv[0] = KM[(size_t)samp * L + k0 + wg * 64 + r0]; kmv[1] = KM[(size_t)samp * L + k0 + wg * 64 + r0 + 8]; }
  float* dbt = DBT + ((size_t)head * L + k0 + wg * 64 + r0) * L;

  float sc[32], dp[32];
  float dsv[2][16];
  uint32_t pp[QB / 16][4], dsr[QB / 16][4];
  const uint32_t sbase = smem_u32(sm);
  for (int n = 0; n < nblocks; ++n) {
    const int sn = n % STAGES;
    const uint32_t slot = sbase + sn * ST_BYTES;
    mbar_wait(&full[sn], (n / STAGES) & 1);
    const int nb = QROT ? (n + samp) % nblocks : n;
    if (!FLOOR) {
      const float* ld = reinterpret_cast<const float*>(sm + sn * ST_BYTES + OFF_LD);
      float lse[16], dd[16];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float2 a = *reinterpret_cast<const float2*>(ld + j * 8 + cb);
        const float2 d = *reinterpret_cast<const float2*>(ld + QB + j * 8 + cb);
        lse[2 * j] = a.x; lse[2 * j + 1] = a.y; dd[2 * j] = d.x; dd[2 * j + 1] = d.y;
      }
      // seed S^T with (bias - LSE)^T: ldmatrix.trans of the [query, key] tile gives each thread its (key, query pair)
      const uint32_t sbt = slot + OFF_BT + wg * SBT;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint32_t bm[2][4];
#pragma unroll
        for (int half = 0; half < 2; ++half)
          ldsm_x4_t(bm[half], sbt + sw128(8 * (4 * half + (lane >> 3)) + (lane & 7), (16 * warp + 8 * h) * 2));
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float2 b = bf2f(bm[j >> 2][j & 3]);
          sc[4 * j + 2 * h] = b.x + (kmv[h] - lse[2 * j]);
          sc[4 * j + 2 * h + 1] = b.y + (kmv[h] - lse[2 * j + 1]);
        }
      }
      wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < DH / 16; ++ks) mma_s_rs(sc, kr[ks], dsw(slot + OFF_QT + ks * 32), 1);
#pragma unroll
      for (int ks = 0; ks < DH / 16; ++ks) mma_s_rs(dp, vr[ks], dsw(slot + OFF_DOT + ks * 32), ks != 0);
      wgmma_commit();
      wgmma_wait<0>();
#pragma unroll
      for (int i = 0; i < 32; ++i) { fence_reg(sc[i]); fence_reg(dp[i]); }
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float p0 = ex2(sc[4 * j + 2 * h]), p1 = ex2(sc[4 * j + 2 * h + 1]);
          const float s0 = p0 * (dp[4 * j + 2 * h] - dd[2 * j]), s1 = p1 * (dp[4 * j + 2 * h + 1] - dd[2 * j + 1]);
          const __nv_bfloat162 pk = __floats2bfloat162_rn(p0, p1), sk = __floats2bfloat162_rn(s0, s1);
          pp[j >> 1][2 * (j & 1) + h] = *reinterpret_cast<const uint32_t*>(&pk);
          dsr[j >> 1][2 * (j & 1) + h] = *reinterpret_cast<const uint32_t*>(&sk);
          if (DBIAS && !RV4) red_v2(dbt + (size_t)(8 * h) * L + nb * QB + j * 8 + cb, s0, s1);
          if (DBIAS && RV4) { dsv[h][2 * j] = s0; dsv[h][2 * j + 1] = s1; }
        }
      if (DBIAS && RV4) {
        // lanes c and c^1 of a quad hold queries 2c, 2c+1 and 2(c^1), 2(c^1)+1 for key rows r0 and r0 + 8: the even
        // lane keeps row r0 and takes its partner's pair of it, the odd lane keeps row r0 + 8 -> 4 adjacent queries
        const bool odd = lane & 1;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float g0 = __shfl_xor_sync(0xffffffffu, odd ? dsv[0][2 * j] : dsv[1][2 * j], 1);
          const float g1 = __shfl_xor_sync(0xffffffffu, odd ? dsv[0][2 * j + 1] : dsv[1][2 * j + 1], 1);
          const int qq = nb * QB + j * 8 + (cb & ~2);                // the quad pair's first query
          if (!odd) red_v4(dbt + qq, dsv[0][2 * j], dsv[0][2 * j + 1], g0, g1);
          else red_v4(dbt + (size_t)8 * L + qq, g0, g1, dsv[1][2 * j], dsv[1][2 * j + 1]);
        }
      }
      wgmma_fence();
#pragma unroll
      for (int i = 0; i < 24; ++i) { fence_reg(dk[i]); fence_reg(dv[i]); }
#pragma unroll
      for (int ks = 0; ks < QB / 16; ++ks) mma_o(dv, pp[ks], dmn(slot + OFF_DOT, ks));
#pragma unroll
      for (int ks = 0; ks < QB / 16; ++ks) mma_o(dk, dsr[ks], dmn(slot + OFF_QT, ks));
      wgmma_commit();
      wgmma_wait<0>();
    } else {
      dk[0] += __int_as_float(*reinterpret_cast<const int*>(sm + sn * ST_BYTES + lane * 4));
    }
    if (lane == 0 && n + STAGES < nblocks) mbar_arrive(&empty[sn]);
  }
#pragma unroll
  for (int i = 0; i < 24; ++i) { fence_reg(dk[i]); fence_reg(dv[i]); }
  constexpr float RL2E = 0.6931471805599453f;                        // 1 / log2 e: q was pre-scaled by log2 e / sqrt 48
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const size_t row = (size_t)samp * L + k0 + wg * 64 + r0 + 8 * h;
    float* pk = DK + row * DM + qcol;
    float* pv = DV + row * DM + qcol;
#pragma unroll
    for (int j = 0; j < 6; ++j) {
      *reinterpret_cast<float2*>(pk + j * 8 + cb) = make_float2(dk[4 * j + 2 * h] * RL2E, dk[4 * j + 2 * h + 1] * RL2E);
      *reinterpret_cast<float2*>(pv + j * 8 + cb) = make_float2(dv[4 * j + 2 * h], dv[4 * j + 2 * h + 1]);
    }
  }
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

// q (pre-scaled), k, v, dO [A*L, 768] bf16; bias [H, L, L] bf16 (log2 units); kmask; LSE, D [A, H, L].
// Writes dK, dV [A*L, 768] fp32 and ADDS dbias^T into DBT [H, L(key), L(query)] fp32 (zero it first).
void attn_dkv(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dO, torch::Tensor bias,
              c10::optional<torch::Tensor> kmask, torch::Tensor LSE, torch::Tensor Dd, torch::Tensor DK, torch::Tensor DV,
              torch::Tensor DBT) {
  const int H = bias.size(0), L = bias.size(1);
  const int A = q.size(0) / L;
  for (auto* t : {&q, &k, &v, &dO})
    TORCH_CHECK(t->is_contiguous() && t->scalar_type() == torch::kBFloat16 && t->size(1) == DM && t->size(0) == A * L, "qkv layout");
  TORCH_CHECK(bias.is_contiguous() && bias.scalar_type() == torch::kBFloat16 && bias.size(2) == L && H * DH == DM, "bias layout");
  for (auto* t : {&DK, &DV, &DBT}) TORCH_CHECK(t->is_contiguous() && t->scalar_type() == torch::kFloat32, "out layout");
  TORCH_CHECK(L % (64 * NWG) == 0 && L % QB == 0, "shape");
  const size_t smem = 1024 + OFF_KV + 2 * SKV + 256;
  const bool hasm = kmask.has_value() && kmask->defined();
  auto kern = hasm ? attn_dkv_kernel<true> : attn_dkv_kernel<false>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  const int nt = L / (64 * NWG);
  auto mq = tile_map(q.data_ptr(), (uint64_t)A * L, DM, DM, QW, QB);
  auto mk = tile_map(k.data_ptr(), (uint64_t)A * L, DM, DM, QW, 64 * NWG);
  auto mv = tile_map(v.data_ptr(), (uint64_t)A * L, DM, DM, QW, 64 * NWG);
  auto mdo = tile_map(dO.data_ptr(), (uint64_t)A * L, DM, DM, QW, QB);
  auto mb = tile_map(bias.data_ptr(), (uint64_t)H * L, L, L, 64, QB);
  kern<<<A * H * nt, 128 * NWG + 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      mq, mk, mv, mdo, mb, hasm ? kmask->data_ptr<float>() : nullptr, LSE.data_ptr<float>(), Dd.data_ptr<float>(),
      DK.data_ptr<float>(), DV.data_ptr<float>(), DBT.data_ptr<float>(), L, H, A, nt);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("attn_dkv", &attn_dkv); }
