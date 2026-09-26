// attn_fwd.cu -- the token DiT attention core's TRAINING forward (sm_90a, bf16 operands, fp32 accumulation):
//
//   S[a] = q[a] k[a]^T / sqrt(48) + bias [+ key mask]     bias [H, L, L] is per head and shared by all A samples
//   O[a] = softmax(S[a]) v[a]                             fp32 out, plus the row log-sum-exp (log2 units) for the backward
//
// Derived from the inference core (token_dit_fused/tdit/cuda_core/attn_core.cu), with its verified pieces kept: the q
// tile held in registers and parked in slot 0's bias area (QDEP release), the bias seeded into the score accumulator,
// the three TMA issuers, NODANGLE. What changes for training: a running max (logits are not bounded while the model
// trains), separate q / k / v tensors, fp32 O and the LSE.
//
// Units: the caller hands q pre-multiplied by log2(e) / sqrt(48), and the bias is seeded times log2(e), so the score
// accumulator holds log2-unit logits and every exponential is one ex2.
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;

#ifndef STAGES
#define STAGES 3
#endif
#ifndef NWG
#define NWG 2                    // consumer warpgroups: query rows per CTA = 64 NWG
#endif
#ifndef BLKSM
#define BLKSM 2
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
#ifndef QWID
#define QWID 48                  // TMA box width for q / k / v / dO: 48 loads exactly the head (the 128-B swizzled smem
#endif                           // image keeps its 128-B row pitch), 64 also pulls 16 columns of the next head
#ifndef FLOOR
#define FLOOR 0                  // 1: the same TMA traffic and barriers, no math -- the measured pattern floor
#endif
constexpr int BN = 64, DH = 48, QW = QWID, QM = 64 * NWG, DM = 768;
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

constexpr int SKV = BN * 128, SB = QM * 128;
constexpr int ST_BYTES = 2 * SKV + SB;
constexpr int QSTAGE = 2 * SKV;                                     // q is parked in slot 0's bias area

// grid: A * H * (L / QM) CTAs, the sample fastest so the A CTAs sharing a bias tile run together and read it from L2
template <bool HASM>
__global__ void __launch_bounds__(128 * NWG + 32, BLKSM)
attn_fwd_kernel(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mk,
                const __grid_constant__ CUtensorMap mv, const __grid_constant__ CUtensorMap mbias,
                const float* __restrict__ KM, float* __restrict__ O, float* __restrict__ LSE, int L, int H, int A, int mt) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + STAGES * ST_BYTES);
  uint64_t* empty = full + STAGES;
  uint64_t* qbar = empty + STAGES;
  uint64_t* qdone = qbar + 1;

  const int tid = threadIdx.x;
  constexpr int PTHR = 32, NTHR = 128 * NWG + PTHR;
  constexpr int LAUNCH_REGS = (65536 / (NTHR * BLKSM)) / 8 * 8;
  constexpr int CONS_RAW = (NTHR * LAUNCH_REGS - PTHR * 40) / (128 * NWG) / 8 * 8;
  constexpr int CONS_REGS = CONS_RAW > 232 ? 232 : CONS_RAW;
  if (RSPLIT) { if (tid >= 128 * NWG) setmaxnreg_dec<40>(); else setmaxnreg_inc<CONS_REGS>(); }

  const int samp = blockIdx.x % A, cid = blockIdx.x / A;
  const int m_tile = cid % mt, head = cid / mt;
  const int m0 = m_tile * QM, qcol = head * DH, row0 = samp * L + m0;
  const int wg = tid >> 7;
  const int nblocks = L / BN;

  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) { mbar_init(&full[s], NISS); mbar_init(&empty[s], FCLS * 4 * NWG); }
    mbar_init(qbar, 1);
    mbar_init(qdone, FCLS * 4 * NWG);                           // cluster-wide: a multicast lands in every rank's slot 0
    fence_barrier_init();
  }
  __syncthreads();
  if (FCLS > 1) cluster_sync_all();                                  // every barrier in the cluster initialised
  const uint32_t rank = FCLS > 1 ? cluster_rank() : 0;

  if (tid >= 128 * NWG) {                                           // producer warp: one issuing thread per tensor
    const int iss = tid - 128 * NWG;                                // 0 = k, 1 = v, 2 = bias
    if (iss == 0) {
      tma_prefetch_desc(&mq); tma_prefetch_desc(&mk); tma_prefetch_desc(&mv); tma_prefetch_desc(&mbias);
      mbar_arrive_expect_tx(qbar, QM * QW * 2);
      tma_load_2d(sm + QSTAGE, &mq, qbar, qcol, row0);
    }
    if (iss < NISS) mbar_wait(qdone, 0);
    if (iss < NISS) {
      for (int n = 0; n < nblocks; ++n) {
        const int s = n % STAGES;
        mbar_wait(&empty[s], ((n / STAGES) & 1) ^ 1);
        uint8_t* slot = sm + s * ST_BYTES;
        if (iss == 0) {
          mbar_arrive_expect_tx(&full[s], BN * QW * 2);
          tma_load_2d(slot, &mk, &full[s], qcol, samp * L + n * BN);
        } else if (iss == 1) {
          mbar_arrive_expect_tx(&full[s], BN * QW * 2);
          tma_load_2d(slot + SKV, &mv, &full[s], qcol, samp * L + n * BN);
        } else {
          if (NOBIAS) { mbar_arrive(&full[s]); continue; }
          mbar_arrive_expect_tx(&full[s], SB);
          if (FCLS == 1) tma_load_2d(slot + 2 * SKV, &mbias, &full[s], n * BN, head * L + m0);
          else if (rank == (uint32_t)(n % FCLS))                     // one rank per block issues, rotating
            tma_load_mc(slot + 2 * SKV, &mbias, &full[s], n * BN, head * L + m0, (1u << FCLS) - 1);
        }
      }
    }
    __syncwarp();
    if (FCLS > 1) cluster_sync_all();                                // no remote arrive may land on an exited CTA
    return;
  }

  // ---- consumer warpgroups: 64 query rows each
  const int lane = tid & 31, warp = (tid >> 5) & 3;
  const int r0 = warp * 16 + (lane >> 2), cb = 2 * (lane & 3);
  float acc[24];
#pragma unroll
  for (int i = 0; i < 24; ++i) acc[i] = 0.f;
  float m_i[2] = {-INFINITY, -INFINITY}, l_i[2] = {0.f, 0.f};
  mbar_wait(qbar, 0);
  const uint32_t sq = smem_u32(sm) + QSTAGE + wg * 8192;
  uint32_t qr[DH / 16][4];
#pragma unroll
  for (int ks = 0; ks < DH / 16; ++ks)
    ldsm_x4(qr[ks], sq + sw128(warp * 16 + 8 * ((lane >> 3) & 1) + (lane & 7), (16 * ks + 8 * (lane >> 4)) * 2));
  {
    uint32_t dep = 0;                                               // the release depends on every q register (QDEP=2)
#pragma unroll
    for (int ks = 0; ks < DH / 16; ++ks) dep ^= qr[ks][0] ^ qr[ks][1] ^ qr[ks][2] ^ qr[ks][3];
    if ((tid & 31) == 0) {
      if (FCLS == 1) mbar_arrive_dep(qdone, zero_dep(dep));
      else for (int r = 0; r < FCLS; ++r) mbar_arrive_remote(reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(qdone) + zero_dep(dep)), r);
    }
  }
  const float* kmr = HASM ? KM + (size_t)samp * L : nullptr;

  float sc[32];
  uint32_t pa[BN / 16][4];
  const uint32_t sbase = smem_u32(sm);
  // Descriptors are built once and moved by adding (byte offset >> 4) to the 14-bit address field (shared addresses
  // stay under 256 KB, so it never carries): smem_desc per k-step cost ~1.8 uniform-datapath instructions a pair.
  const uint64_t dK0 = dsw(sbase), dV0 = dmn(sbase + SKV, 0);
  for (int n = 0; n < nblocks; ++n) {
    const int sn = n % STAGES;
    const uint32_t slot = sbase + sn * ST_BYTES;
    mbar_wait(&full[sn], (n / STAGES) & 1);
    if (!FLOOR) {
      // seed with bias * log2 e (+ the key mask), the QK wgmma accumulates onto it
      float mk[16];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (HASM) { const float2 t = *reinterpret_cast<const float2*>(kmr + n * BN + j * 8 + cb); mk[2 * j] = t.x; mk[2 * j + 1] = t.y; }
        else { mk[2 * j] = 0.f; mk[2 * j + 1] = 0.f; }
      }
      const uint32_t sbn = slot + 2 * SKV + wg * 8192;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint32_t bm[2][4];
#pragma unroll
        for (int half = 0; half < 2; ++half)
          ldsm_x4(bm[half], sbn + sw128(16 * warp + 8 * h + (lane & 7), 8 * (4 * half + (lane >> 3)) * 2));
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float2 b = bf2f(bm[j >> 2][j & 3]);
          sc[4 * j + 2 * h] = HASM ? b.x + mk[2 * j] : b.x;          // x + 0.f is not foldable (signed zero)
          sc[4 * j + 2 * h + 1] = HASM ? b.y + mk[2 * j + 1] : b.y;
        }
      }
      wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < DH / 16; ++ks) mma_s_rs(sc, qr[ks], dK0 + ((sn * ST_BYTES + ks * 32) >> 4), 1);
      wgmma_commit();
      wgmma_wait<0>();
#pragma unroll
      for (int i = 0; i < 32; ++i) fence_reg(sc[i]);
      // Lazy running max (FA4): the max only moves when a block exceeds it by more than LAZY (log2 units), so p stays
      // under 2^LAZY and the accumulator rescale -- 24 FMULs a thread -- runs only in warps where some row moved.
      float m_new[2];
      bool moved = false;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float mx = -INFINITY;
#pragma unroll
        for (int j = 0; j < 8; ++j) mx = fmaxf(mx, fmaxf(sc[4 * j + 2 * h], sc[4 * j + 2 * h + 1]));
        mx = qmax(mx);
        m_new[h] = mx > m_i[h] + LAZY ? mx : m_i[h];
        moved |= m_new[h] != m_i[h];
      }
      if (__any_sync(0xffffffffu, moved)) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float alpha = ex2(m_i[h] - m_new[h]);               // m_i = -inf on the first block: alpha = 0
          l_i[h] *= alpha;
#pragma unroll
          for (int j = 0; j < 6; ++j) { acc[4 * j + 2 * h] *= alpha; acc[4 * j + 2 * h + 1] *= alpha; }
          m_i[h] = m_new[h];
        }
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float ssum = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float p0 = ex2(sc[4 * j + 2 * h] - m_i[h]), p1 = ex2(sc[4 * j + 2 * h + 1] - m_i[h]);
          ssum += p0 + p1;
          const __nv_bfloat162 pk = __floats2bfloat162_rn(p0, p1);
          pa[j >> 1][2 * (j & 1) + h] = *reinterpret_cast<const uint32_t*>(&pk);
        }
        l_i[h] += ssum;                                             // per-thread partial; the quad sum is in the epilogue
      }
      wgmma_fence();
#pragma unroll
      for (int i = 0; i < 24; ++i) fence_reg(acc[i]);
#pragma unroll
      for (int ks = 0; ks < BN / 16; ++ks) mma_o(acc, pa[ks], dV0 + ((sn * ST_BYTES + ks * 2048) >> 4));
      wgmma_commit();
      wgmma_wait<0>();
    } else {
      acc[0] += __int_as_float(*reinterpret_cast<const int*>(sm + sn * ST_BYTES + (tid & 31) * 4));
    }
    if ((tid & 31) == 0 && n + STAGES < nblocks) {
      if (FCLS == 1) mbar_arrive(&empty[sn]);
      else for (int r = 0; r < FCLS; ++r) mbar_arrive_remote(&empty[sn], r);   // the slot is free in every rank's view
    }
  }
#pragma unroll
  for (int i = 0; i < 24; ++i) fence_reg(acc[i]);

  // ---- epilogue: O = acc / l (fp32), LSE = m + log2 l
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const float l = qsum(l_i[h]);
    const float inv = 1.f / l;
    const int row = m0 + wg * 64 + r0 + 8 * h;
    float* orow = O + (size_t)(samp * L + row) * DM + qcol;
#pragma unroll
    for (int j = 0; j < 6; ++j)
      *reinterpret_cast<float2*>(orow + j * 8 + cb) = make_float2(acc[4 * j + 2 * h] * inv, acc[4 * j + 2 * h + 1] * inv);
    if ((lane & 3) == 0) LSE[((size_t)samp * H + head) * L + row] = m_i[h] + __log2f(l);
  }
  if (FCLS > 1) { __syncwarp(); cluster_sync_all(); }
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

// q, k, v [A*L, 768] bf16 (q pre-scaled by log2 e / sqrt 48); bias [H, L, L] bf16; kmask [A, L] fp32 additive (log2
// units) or empty. Writes O [A*L, 768] fp32 and LSE [A, H, L] fp32 (log2 units).
void attn_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bias, c10::optional<torch::Tensor> kmask,
              torch::Tensor O, torch::Tensor LSE) {
  const int H = bias.size(0), L = bias.size(1);
  const int A = q.size(0) / L;
  for (auto* t : {&q, &k, &v})
    TORCH_CHECK(t->is_contiguous() && t->scalar_type() == torch::kBFloat16 && t->size(1) == DM && t->size(0) == A * L, "qkv layout");
  TORCH_CHECK(bias.is_contiguous() && bias.scalar_type() == torch::kBFloat16 && bias.size(2) == L && H * DH == DM, "bias layout");
  TORCH_CHECK(O.is_contiguous() && O.scalar_type() == torch::kFloat32 && LSE.is_contiguous(), "out layout");
  TORCH_CHECK(L % QM == 0, "L must be a multiple of the query tile");
  const size_t smem = 1024 + STAGES * ST_BYTES + 256;
  const bool hasm = kmask.has_value() && kmask->defined();
  auto kern = hasm ? attn_fwd_kernel<true> : attn_fwd_kernel<false>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  const int mt = L / QM;
  auto mq = tile_map(q.data_ptr(), (uint64_t)A * L, DM, DM, QW, QM);
  auto mk = tile_map(k.data_ptr(), (uint64_t)A * L, DM, DM, QW, BN);
  auto mv = tile_map(v.data_ptr(), (uint64_t)A * L, DM, DM, QW, BN);
  auto mb = tile_map(bias.data_ptr(), (uint64_t)H * L, L, L, BN, QM);
  TORCH_CHECK(A % FCLS == 0, "A must be a multiple of the cluster size");
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(A * H * mt);
  cfg.blockDim = dim3(128 * NWG + 32);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = FCLS; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, kern, mq, mk, mv, mb, hasm ? kmask->data_ptr<float>() : nullptr, O.data_ptr<float>(),
                                 LSE.data_ptr<float>(), L, H, A, mt) == cudaSuccess, "launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("attn_fwd", &attn_fwd); }
