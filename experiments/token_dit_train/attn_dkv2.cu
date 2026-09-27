// attn_dkv2.cu -- the backward's dK / dV / dbias pass (K1) with dbias summed over the samples ON CHIP.
//
// attn_dkv.cu (a CTA per sample) needs the sum over the A samples for dbias, and L2 atomics price it at ~700 us at L768.
// Here a CTA owns a dbias region -- QR = 192 queries x 128 keys (2 warpgroups x 64 keys) -- keeps it in shared memory
// (fp32, 96 KB) with the bias of the same region (bf16, 48 KB), and loops over a chunk of samples:
//
//   per sample: K, V (128 keys) -> registers; for each of the 3 query blocks: Q, dO, LSE, D stream through a TMA ring
//     S^T = K Q^T + (bias - LSE)^T, dP^T = V dO^T, P^T = 2^S^T, dS^T = P^T (dP^T - D)
//     dV += P^T dO, dK += dS^T Q, dbias_smem += dS^T
//   dK / dV are partial over the CTA's QR queries: the KS = L / QR CTAs of a cluster (one per query range) add them into
//   the owning CTA's shared memory (red.shared::cluster), which writes them out.
//
// At the end of the chunk the CTA writes its dbias region (transposed, [key, query]) into its chunk's partial.
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;

#ifndef STAGES
#define STAGES 3
#endif
#ifndef XCH
#define XCH 1                    // timing diagnostics: 0 skips the dK / dV exchange (wrong results)
#endif
#ifndef DBS
#define DBS 1                    // timing diagnostics: 0 skips the dbias accumulation (wrong results)
#endif
constexpr int NWG = 2, DH = 48, QW = 64, DM = 768, QB = 64, QR = 192, NQB = QR / QB, KT = 64 * NWG;

TMN_DEVI uint64_t dsw(uint32_t addr) { return smem_desc(addr, 16, 1024, 1); }
TMN_DEVI uint64_t dmn(uint32_t base, int ks) { return smem_desc(base + ks * 2048, 16, 1024, 1); }
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
TMN_DEVI uint32_t mapa(uint32_t a, uint32_t rank) {
  uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank)); return r;
}
TMN_DEVI void red_cl(uint32_t caddr, float v) { asm volatile("red.shared::cluster.add.f32 [%0], %1;" :: "r"(caddr), "f"(v) : "memory"); }
TMN_DEVI void arrive_rel_cl(uint32_t cbar) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(cbar) : "memory");
}
TMN_DEVI void wait_acq_cl(uint64_t* bar, uint32_t phase) {
  uint32_t ok;
  do {
    asm volatile("{ .reg .pred P; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P, [%1], %2; selp.u32 %0, 1, 0, P; }"
                 : "=r"(ok) : "r"(smem_u32(bar)), "r"(phase) : "memory");
  } while (!ok);
}
TMN_DEVI void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
TMN_DEVI uint32_t cluster_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r)); return r; }
TMN_DEVI float2 bf2f(uint32_t u) { return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u)); }

// shared memory
constexpr int SQT = QB * 128;                                       // a 64-row, 64-wide bf16 tile
constexpr int OFF_DB = 0, SZ_DB = NWG * NQB * 32 * 128 * 4;         // dbias, fragment order: [wg][qblock][i][thread]
constexpr int OFF_BI = OFF_DB + SZ_DB, SZ_BI = NQB * NWG * SQT;     // resident bias tiles [qblock][wg] (64 q x 64 keys)
constexpr int OFF_LD = 2 * SQT;                                     // in a query slot: Q, dO, then LSE and D
constexpr int SLOT = (OFF_LD + 2 * QB * 4 + 1023) / 1024 * 1024;    // (a K or V slot holds 128 rows = 16 KB <= SLOT)
constexpr int OFF_RG = OFF_BI + SZ_BI;
constexpr int OFF_X = OFF_RG + STAGES * SLOT;                       // dK / dV pieces this CTA owns: up to 2 x 12 KB
constexpr int SZ_PC = 24 * 128 * 4;
constexpr int OFF_BAR = OFF_X + 2 * SZ_PC;
static_assert(2 * SQT <= SLOT && KT * 128 <= SLOT, "slot");

// grid: KS * (L / KT) * H * nch CTAs, cluster = the KS query ranges of one (key tile, head, chunk)
template <bool HASM, int KS>
__global__ void __launch_bounds__(128 * NWG + 32, 1)
attn_dkv2_kernel(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mk,
                 const __grid_constant__ CUtensorMap mv, const __grid_constant__ CUtensorMap mdo,
                 const __grid_constant__ CUtensorMap mbias, const float* __restrict__ KM, const float* __restrict__ LSE,
                 const float* __restrict__ DD, float* __restrict__ DK, float* __restrict__ DV,
                 float* __restrict__ DBT, int L, int H, int sch) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint64_t* bbar = reinterpret_cast<uint64_t*>(sm + OFF_BAR);
  uint64_t* full = bbar + 1;
  uint64_t* empty = full + STAGES;
  uint64_t* abar = empty + STAGES;                                  // [2]: all KS partials of an owned piece are in
  uint64_t* zbar = abar + 2;                                        // the owners have drained the previous sample

  const int tid = threadIdx.x;
  const uint32_t rank = KS > 1 ? cluster_rank() : 0;
  int bid = blockIdx.x / KS;
  const int nkt = L / KT;
  const int kt = bid % nkt; bid /= nkt;
  const int head = bid % H, chunk = bid / H;
  const int k0 = kt * KT, q0 = rank * QR, qcol = head * DH;
  const int wg = tid >> 7;
  static_assert(KS == 2 || KS == 4, "the owned-piece buffers hold 4 / KS pieces");

  if (tid == 0) {
    mbar_init(bbar, 1);
    for (int s = 0; s < STAGES; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], 4 * NWG); }
    for (int p = 0; p < 2; ++p) mbar_init(&abar[p], KS * 128);
    mbar_init(zbar, 4 * 128);
    fence_barrier_init();
  }
  // the owned pieces start at zero
  for (int i = tid; i < 2 * SZ_PC / 16; i += blockDim.x) reinterpret_cast<float4*>(sm + OFF_X)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int i = tid; i < SZ_DB / 16; i += blockDim.x) reinterpret_cast<float4*>(sm + OFF_DB)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
  __syncthreads();
  if (KS > 1) cluster_sync_all();

  if (tid >= 128 * NWG) {                                           // producer: one thread
    if (tid == 128 * NWG) {
      tma_prefetch_desc(&mq); tma_prefetch_desc(&mk); tma_prefetch_desc(&mv); tma_prefetch_desc(&mdo);
      tma_prefetch_desc(&mbias);
      mbar_arrive_expect_tx(bbar, SZ_BI);
      for (int b = 0; b < NQB; ++b)
        for (int w = 0; w < NWG; ++w)
          tma_load_2d(sm + OFF_BI + (b * NWG + w) * SQT, &mbias, bbar, k0 + 64 * w, head * L + q0 + b * QB);
      int t = 0;
      auto next = [&]() -> uint8_t* {
        const int s = t % STAGES;
        mbar_wait(&empty[s], ((t / STAGES) & 1) ^ 1);
        return sm + OFF_RG + s * SLOT;
      };
      for (int i = 0; i < sch; ++i) {
        const int a = chunk * sch + i;
        uint8_t* sl = next();
        mbar_arrive_expect_tx(&full[t % STAGES], KT * 128);
        tma_load_2d(sl, &mk, &full[t % STAGES], qcol, a * L + k0); ++t;
        sl = next();
        mbar_arrive_expect_tx(&full[t % STAGES], KT * 128);
        tma_load_2d(sl, &mv, &full[t % STAGES], qcol, a * L + k0); ++t;
        for (int b = 0; b < NQB; ++b, ++t) {
          sl = next();
          uint64_t* f = &full[t % STAGES];
          mbar_arrive_expect_tx(f, OFF_LD + 2 * QB * 4);
          tma_load_2d(sl, &mq, f, qcol, a * L + q0 + b * QB);
          tma_load_2d(sl + SQT, &mdo, f, qcol, a * L + q0 + b * QB);
          const size_t li = ((size_t)a * H + head) * L + q0 + b * QB;
          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                       :: "r"(smem_u32(sl + OFF_LD)), "l"(LSE + li), "r"(QB * 4), "r"(smem_u32(f)) : "memory");
          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                       :: "r"(smem_u32(sl + OFF_LD + QB * 4)), "l"(DD + li), "r"(QB * 4), "r"(smem_u32(f)) : "memory");
        }
      }
    }
    __syncwarp();
    if (KS > 1) cluster_sync_all();
    return;
  }

  const int lane = tid & 31, warp = (tid >> 5) & 3, wt = tid & 127;
  const int r0 = warp * 16 + (lane >> 2), cb = 2 * (lane & 3);
  const uint32_t sbase = smem_u32(sm);
  // the pieces: p = 2 * kind + wg (kind 0 = dK, 1 = dV), owned by rank p % KS at local index p / KS
  const int pk = wg, pv = 2 + wg;
  const uint32_t ok_rank = pk % KS, ov_rank = pv % KS;
  const uint32_t xk = mapa(sbase + OFF_X + (pk / KS) * SZ_PC, ok_rank), xv = mapa(sbase + OFF_X + (pv / KS) * SZ_PC, ov_rank);
  const uint32_t ak = mapa(smem_u32(&abar[pk / KS]), ok_rank), av = mapa(smem_u32(&abar[pv / KS]), ov_rank);
  mbar_wait(bbar, 0);

  int t = 0;
  float sc[32], dp[32];
  uint32_t pp[QB / 16][4], dsr[QB / 16][4];
  for (int i = 0; i < sch; ++i) {
    const int a = chunk * sch + i;
    uint32_t kr[DH / 16][4], vr[DH / 16][4];
#pragma unroll
    for (int kv = 0; kv < 2; ++kv, ++t) {                           // K, then V: into registers, the slot released
      const int s = t % STAGES;
      mbar_wait(&full[s], (t / STAGES) & 1);
      const uint32_t base = sbase + OFF_RG + s * SLOT + wg * 8192;
      uint32_t dep = 0;
#pragma unroll
      for (int ks = 0; ks < DH / 16; ++ks) {
        const int rr = warp * 16 + 8 * ((lane >> 3) & 1) + (lane & 7), bb = (16 * ks + 8 * (lane >> 4)) * 2;
        if (kv == 0) { ldsm_x4(kr[ks], base + sw128(rr, bb)); dep ^= kr[ks][0] ^ kr[ks][1] ^ kr[ks][2] ^ kr[ks][3]; }
        else { ldsm_x4(vr[ks], base + sw128(rr, bb)); dep ^= vr[ks][0] ^ vr[ks][1] ^ vr[ks][2] ^ vr[ks][3]; }
      }
      if (lane == 0) mbar_arrive_dep(&empty[s], zero_dep(dep));
    }
    float kmv[2] = {0.f, 0.f};
    if (HASM) { kmv[0] = KM[(size_t)a * L + k0 + wg * 64 + r0]; kmv[1] = KM[(size_t)a * L + k0 + wg * 64 + r0 + 8]; }
    float dk[24], dv[24];
#pragma unroll
    for (int j = 0; j < 24; ++j) { dk[j] = 0.f; dv[j] = 0.f; }

    for (int b = 0; b < NQB; ++b, ++t) {
      const int s = t % STAGES;
      const uint32_t slot = sbase + OFF_RG + s * SLOT;
      mbar_wait(&full[s], (t / STAGES) & 1);
      const float* ld = reinterpret_cast<const float*>(sm + OFF_RG + s * SLOT + OFF_LD);
      const uint32_t sbt = sbase + OFF_BI + (b * NWG + wg) * SQT;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint32_t bm[2][4];
#pragma unroll
        for (int half = 0; half < 2; ++half)
          ldsm_x4_t(bm[half], sbt + sw128(8 * (4 * half + (lane >> 3)) + (lane & 7), (16 * warp + 8 * h) * 2));
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float2 bb = bf2f(bm[j >> 2][j & 3]);
          const float2 ls = *reinterpret_cast<const float2*>(ld + j * 8 + cb);
          sc[4 * j + 2 * h] = bb.x + (kmv[h] - ls.x);
          sc[4 * j + 2 * h + 1] = bb.y + (kmv[h] - ls.y);
        }
      }
      wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < DH / 16; ++ks) mma_s_rs(sc, kr[ks], dsw(slot + ks * 32), 1);
#pragma unroll
      for (int ks = 0; ks < DH / 16; ++ks) mma_s_rs(dp, vr[ks], dsw(slot + SQT + ks * 32), ks != 0);
      wgmma_commit();
      wgmma_wait<0>();
#pragma unroll
      for (int j = 0; j < 32; ++j) { fence_reg(sc[j]); fence_reg(dp[j]); }
      float4* db = reinterpret_cast<float4*>(sm + OFF_DB) + ((wg * NQB + b) * 8) * 128 + wt;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float2 d = *reinterpret_cast<const float2*>(ld + QB + j * 8 + cb);
        float sv[4];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float p0 = ex2(sc[4 * j + 2 * h]), p1 = ex2(sc[4 * j + 2 * h + 1]);
          const float s0 = p0 * (dp[4 * j + 2 * h] - d.x), s1 = p1 * (dp[4 * j + 2 * h + 1] - d.y);
          const __nv_bfloat162 pk2 = __floats2bfloat162_rn(p0, p1), sk2 = __floats2bfloat162_rn(s0, s1);
          pp[j >> 1][2 * (j & 1) + h] = *reinterpret_cast<const uint32_t*>(&pk2);
          dsr[j >> 1][2 * (j & 1) + h] = *reinterpret_cast<const uint32_t*>(&sk2);
          sv[2 * h] = s0; sv[2 * h + 1] = s1;
        }
        if (DBS) {
          float4 o = db[j * 128];                                   // this thread's own dbias values: no conflicts
          o.x += sv[0]; o.y += sv[1]; o.z += sv[2]; o.w += sv[3];
          db[j * 128] = o;
        }
      }
      wgmma_fence();
#pragma unroll
      for (int j = 0; j < 24; ++j) { fence_reg(dk[j]); fence_reg(dv[j]); }
#pragma unroll
      for (int ks = 0; ks < QB / 16; ++ks) mma_o(dv, pp[ks], dmn(slot + SQT, ks));
#pragma unroll
      for (int ks = 0; ks < QB / 16; ++ks) mma_o(dk, dsr[ks], dmn(slot, ks));
      wgmma_commit();
      wgmma_wait<0>();
      if (lane == 0) mbar_arrive(&empty[s]);
    }
#pragma unroll
    for (int j = 0; j < 24; ++j) { fence_reg(dk[j]); fence_reg(dv[j]); }

    // ---- dK / dV: add this range's partial into the owners' pieces
    if (!XCH) {
      if (rank == 0) {
        float* out = DK + ((size_t)a * L + k0 + wg * 64) * DM + qcol;
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int j = 0; j < 6; ++j)
            *reinterpret_cast<float2*>(out + (size_t)(r0 + 8 * h) * DM + j * 8 + cb) = make_float2(dk[4 * j + 2 * h] + dv[4 * j + 2 * h], dk[4 * j + 2 * h + 1]);
      }
      continue;
    }
    if (i > 0) wait_acq_cl(zbar, (i - 1) & 1);                      // the owners drained sample i - 1
#pragma unroll
    for (int j = 0; j < 24; ++j) { red_cl(xk + (j * 128 + wt) * 4, dk[j]); red_cl(xv + (j * 128 + wt) * 4, dv[j]); }
    arrive_rel_cl(ak); arrive_rel_cl(av);
    // ---- owner: drain the pieces this warpgroup owns
#pragma unroll
    for (int kind = 0; kind < 2; ++kind) {
      const int p = 2 * kind + wg;
      if (p % KS != (int)rank) continue;
      uint64_t* ab = &abar[p / KS];
      wait_acq_cl(ab, i & 1);
      float* x = reinterpret_cast<float*>(sm + OFF_X + (p / KS) * SZ_PC);
      const float sc_ = kind == 0 ? 0.6931471805599453f : 1.f;       // dK: q was pre-scaled by log2 e / sqrt 48
      float* out = (kind == 0 ? DK : DV) + ((size_t)a * L + k0 + wg * 64) * DM + qcol;
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int j = 0; j < 6; ++j) {
          float* e0 = x + ((4 * j + 2 * h) * 128 + wt);
          float* e1 = x + ((4 * j + 2 * h + 1) * 128 + wt);
          *reinterpret_cast<float2*>(out + (size_t)(r0 + 8 * h) * DM + j * 8 + cb) = make_float2(*e0 * sc_, *e1 * sc_);
          *e0 = 0.f; *e1 = 0.f;
        }
      for (int r = 0; r < KS; ++r) arrive_rel_cl(mapa(smem_u32(zbar), r));
    }
  }

  // ---- this chunk's dbias region, transposed: DBT[chunk][head][key][query]
  float* dbt = DBT + (((size_t)chunk * H + head) * L + k0 + wg * 64 + r0) * L + q0;
#pragma unroll 1
  for (int b = 0; b < NQB; ++b) {
    const float4* db = reinterpret_cast<const float4*>(sm + OFF_DB) + ((wg * NQB + b) * 8) * 128 + wt;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float4 o = db[j * 128];
      *reinterpret_cast<float2*>(dbt + b * QB + j * 8 + cb) = make_float2(o.x, o.y);
      *reinterpret_cast<float2*>(dbt + (size_t)8 * L + b * QB + j * 8 + cb) = make_float2(o.z, o.w);
    }
  }
  if (KS > 1) { __syncwarp(); cluster_sync_all(); }
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
template <bool HASM, int KS>
void launch(const CUtensorMap* m, const float* km, const float* lse, const float* dd, float* dk, float* dv, float* dbt,
            int L, int H, int nch, int sch) {
  const size_t smem = 1024 + OFF_BAR + 256;
  auto kern = attn_dkv2_kernel<HASM, KS>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  if (KS > 4) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(KS * (L / KT) * H * nch);
  cfg.blockDim = dim3(128 * NWG + 32);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = KS; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, kern, m[0], m[1], m[2], m[3], m[4], km, lse, dd, dk, dv, dbt, L, H, sch)
              == cudaSuccess, "launch failed");
}
}  // namespace

// as attn_dkv.cu, plus sch (samples per CTA, divides A). DBT [A / sch, H, L(key), L(query)] fp32: each chunk's dbias^T,
// written (not added); the caller sums the chunks.
void attn_dkv2(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dO, torch::Tensor bias,
               c10::optional<torch::Tensor> kmask, torch::Tensor LSE, torch::Tensor Dd, torch::Tensor DK, torch::Tensor DV,
               torch::Tensor DBT, int64_t sch) {
  const int H = bias.size(0), L = bias.size(1);
  const int A = q.size(0) / L;
  for (auto* t : {&q, &k, &v, &dO})
    TORCH_CHECK(t->is_contiguous() && t->scalar_type() == torch::kBFloat16 && t->size(1) == DM && t->size(0) == A * L, "qkv layout");
  TORCH_CHECK(bias.is_contiguous() && bias.scalar_type() == torch::kBFloat16 && bias.size(2) == L && H * DH == DM, "bias layout");
  for (auto* t : {&DK, &DV, &DBT}) TORCH_CHECK(t->is_contiguous() && t->scalar_type() == torch::kFloat32, "out layout");
  const int KS = L / QR;
  TORCH_CHECK(L % QR == 0 && L % KT == 0 && (KS == 2 || KS == 4) && A % sch == 0, "shape");
  TORCH_CHECK(DBT.numel() == (int64_t)(A / sch) * H * L * L, "DBT size");
  const bool hasm = kmask.has_value() && kmask->defined();
  CUtensorMap m[5] = {tile_map(q.data_ptr(), (uint64_t)A * L, DM, DM, QW, QB), tile_map(k.data_ptr(), (uint64_t)A * L, DM, DM, QW, KT),
                      tile_map(v.data_ptr(), (uint64_t)A * L, DM, DM, QW, KT), tile_map(dO.data_ptr(), (uint64_t)A * L, DM, DM, QW, QB),
                      tile_map(bias.data_ptr(), (uint64_t)H * L, L, L, 64, QB)};
  const float* km = hasm ? kmask->data_ptr<float>() : nullptr;
  auto args = std::make_tuple(km, LSE.data_ptr<float>(), Dd.data_ptr<float>(), DK.data_ptr<float>(), DV.data_ptr<float>(),
                              DBT.data_ptr<float>());
  const int nch = A / sch;
#define L_(HM, K) launch<HM, K>(m, std::get<0>(args), std::get<1>(args), std::get<2>(args), std::get<3>(args), \
                                std::get<4>(args), std::get<5>(args), L, H, nch, (int)sch)
  if (hasm) { if (KS == 2) L_(true, 2); else L_(true, 4); }
  else { if (KS == 2) L_(false, 2); else L_(false, 4); }
#undef L_
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("attn_dkv2", &attn_dkv2); }
