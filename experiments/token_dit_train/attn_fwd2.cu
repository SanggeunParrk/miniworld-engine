// attn_fwd2.cu -- training forward with the bias RESIDENT in shared memory and the samples looped inside the CTA.
//
// attn_fwd.cu streams K, V and the bias per (sample, head, query tile): 4 bytes into the SM per (query, key) pair at 128
// query rows, and its no-math pattern alone takes 304 us at L768 / A48 (the math build 355). A cluster multicast of the
// bias makes that pattern SLOWER (324-341 us), so the cost is the bytes each SM takes in, not L2 reads. The bias is the
// same for every sample, so here a CTA keeps its bias rows for a range of KR = 384 keys in shared memory (128 x 384 bf16
// = 96 KB, loaded once) and loops over a chunk of SCH samples: the bias costs 2 / SCH bytes a pair instead of 2.
//
// At L = 768 the keys are split over a 2-CTA cluster (KS = L / KR). Each CTA runs the online softmax over its half and
// the halves meet through DSMEM: CTA r owns warpgroup r's rows; the other warpgroup sends (acc, m, l) to the partner
// with st.async (completing a tx mbarrier there), and the owner combines and writes O and the LSE.
//
//   S = q k^T / sqrt 48 + bias [+ key mask]   O = softmax(S) v   (log2 units inside, as attn_fwd.cu)
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;

#ifndef STAGES
#define STAGES 4                 // K / V ring depth
#endif
#ifndef FLOOR
#define FLOOR 0
#endif
#ifndef PIPE
#define PIPE 1                   // intra-warpgroup pipeline (FLOOR needs PIPE=0)
#endif
constexpr int NWG = 2, BN = 64, DH = 48, QW = 64, QM = 64 * NWG, DM = 768, KR = 384, NKB = KR / BN;
constexpr float LOG2E = 1.4426950408889634f;

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
TMN_DEVI void mbar_arrive_remote(uint64_t* bar, uint32_t rank) {
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(mapa(smem_u32(bar), rank)) : "memory");
}
TMN_DEVI void st_async4(uint32_t caddr, float a, float b, float c, float d, uint32_t cbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               :: "r"(caddr), "f"(a), "f"(b), "f"(c), "f"(d), "r"(cbar) : "memory");
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

// shared memory: resident bias (NKB tiles of QM x 64), the K/V ring, the q ring (2), the exchange buffers (2), barriers
constexpr int SKV = BN * 128, SB = QM * 128, SQ = QM * 128;
constexpr int XFL = 28;                                             // floats a thread sends: acc 24, m 2, l 2
constexpr int SX = 128 * XFL * 4;
constexpr int OFF_B = 0, OFF_KV = OFF_B + NKB * SB, OFF_Q = OFF_KV + STAGES * 2 * SKV, OFF_X = OFF_Q + 2 * SQ;
constexpr int OFF_BAR = OFF_X + 2 * SX;

// grid: KS * mt * nch * H CTAs, cluster = the KS key ranges of one (head, query tile, sample chunk)
template <bool HASM, int KS>
__global__ void __launch_bounds__(128 * NWG + 32, 1)
attn_fwd2_kernel(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mk,
                 const __grid_constant__ CUtensorMap mv, const __grid_constant__ CUtensorMap mbias,
                 const float* __restrict__ KM, float* __restrict__ O, float* __restrict__ LSE, int L, int H, int mt,
                 int sch) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint64_t* bfull = reinterpret_cast<uint64_t*>(sm + OFF_BAR);
  uint64_t* kfull = bfull + 1;
  uint64_t* kempty = kfull + STAGES;
  uint64_t* qfull = kempty + STAGES;
  uint64_t* qempty = qfull + 2;
  uint64_t* xfull = qempty + 2;
  uint64_t* xempty = xfull + 2;

  const int tid = threadIdx.x;
  const uint32_t rank = KS > 1 ? cluster_rank() : 0;
  int bid = blockIdx.x / KS;
  const int m_tile = bid % mt; bid /= mt;
  const int nch = gridDim.x / (KS * mt * H);
  const int chunk = bid % nch, head = bid / nch;
  const int m0 = m_tile * QM, qcol = head * DH, kr0 = rank * KR;
  const int wg = tid >> 7;

  if (tid == 0) {
    mbar_init(bfull, 1);
    for (int s = 0; s < STAGES; ++s) { mbar_init(&kfull[s], 2); mbar_init(&kempty[s], 4 * NWG); }
    for (int s = 0; s < 2; ++s) {
      mbar_init(&qfull[s], 1); mbar_init(&qempty[s], 4 * NWG);
      mbar_init(&xfull[s], 1); mbar_init(&xempty[s], 4);
    }
    fence_barrier_init();
  }
  __syncthreads();
  if (KS > 1) cluster_sync_all();

  if (tid >= 128 * NWG) {                                           // producer warp
    const int iss = tid - 128 * NWG;                                // 0: bias then q, 1: K, 2: V
    if (iss == 0) {
      tma_prefetch_desc(&mq); tma_prefetch_desc(&mk); tma_prefetch_desc(&mv); tma_prefetch_desc(&mbias);
      mbar_arrive_expect_tx(bfull, NKB * SB);
      for (int n = 0; n < NKB; ++n) tma_load_2d(sm + OFF_B + n * SB, &mbias, bfull, kr0 + n * BN, head * L + m0);
      for (int i = 0; i < sch; ++i) {
        const int b = i & 1, a = chunk * sch + i;
        mbar_wait(&qempty[b], ((i >> 1) & 1) ^ 1);
        mbar_arrive_expect_tx(&qfull[b], SQ);
        tma_load_2d(sm + OFF_Q + b * SQ, &mq, &qfull[b], qcol, a * L + m0);
      }
    } else if (iss <= 2) {
      const CUtensorMap* map = iss == 1 ? &mk : &mv;
      for (int i = 0, t = 0; i < sch; ++i) {
        const int a = chunk * sch + i;
        for (int n = 0; n < NKB; ++n, ++t) {
          const int s = t % STAGES;
          mbar_wait(&kempty[s], ((t / STAGES) & 1) ^ 1);
          mbar_arrive_expect_tx(&kfull[s], SKV);
          tma_load_2d(sm + OFF_KV + s * 2 * SKV + (iss - 1) * SKV, map, &kfull[s], qcol, a * L + kr0 + n * BN);
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
  const bool owner = KS == 1 || wg == (int)rank;                    // this warpgroup's rows are finished in this CTA
  mbar_wait(bfull, 0);
  float sc[32];
  uint32_t pa[BN / 16][4];

  for (int i = 0, t = 0; i < sch; ++i) {
    const int a = chunk * sch + i, qb = i & 1;
    uint32_t qr[DH / 16][4];
    mbar_wait(&qfull[qb], (i >> 1) & 1);
    {
      const uint32_t sq = sbase + OFF_Q + qb * SQ + wg * 8192;
#pragma unroll
      for (int ks = 0; ks < DH / 16; ++ks)
        ldsm_x4(qr[ks], sq + sw128(warp * 16 + 8 * ((lane >> 3) & 1) + (lane & 7), (16 * ks + 8 * (lane >> 4)) * 2));
      uint32_t dep = 0;
#pragma unroll
      for (int ks = 0; ks < DH / 16; ++ks) dep ^= qr[ks][0] ^ qr[ks][1] ^ qr[ks][2] ^ qr[ks][3];
      if (lane == 0) mbar_arrive_dep(&qempty[qb], zero_dep(dep));
    }
    const float* kmr = HASM ? KM + (size_t)a * L + kr0 : nullptr;
    float acc[24];
#pragma unroll
    for (int j = 0; j < 24; ++j) acc[j] = 0.f;
    float m_i[2] = {-INFINITY, -INFINITY}, l_i[2] = {0.f, 0.f};

#if PIPE
    // Intra-warpgroup pipeline (FA3): QK(n+1) is issued before softmax(n), and softmax(n) runs while PV(n-1) is still
    // in the tensor pipe. Commit order per iteration n: QK(n+1), then PV(n). Two named score buffers, the loop unrolled
    // by two, so no buffer is indexed dynamically (ptxas would serialise the pipeline, C7514).
    auto seed_qk = [&](float* dst, int n, int tt) {
      const int s = tt % STAGES;
      const uint32_t kslot = sbase + OFF_KV + s * 2 * SKV;
      mbar_wait(&kfull[s], (tt / STAGES) & 1);
      float mk[16];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (HASM) { const float2 v = *reinterpret_cast<const float2*>(kmr + n * BN + j * 8 + cb); mk[2 * j] = v.x; mk[2 * j + 1] = v.y; }
        else { mk[2 * j] = 0.f; mk[2 * j + 1] = 0.f; }
      }
      const uint32_t sbn = sbase + OFF_B + n * SB + wg * 8192;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint32_t bm[2][4];
#pragma unroll
        for (int half = 0; half < 2; ++half)
          ldsm_x4(bm[half], sbn + sw128(16 * warp + 8 * h + (lane & 7), 8 * (4 * half + (lane >> 3)) * 2));
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float2 b = bf2f(bm[j >> 2][j & 3]);
          dst[4 * j + 2 * h] = b.x + (mk[2 * j]);
          dst[4 * j + 2 * h + 1] = b.y + (mk[2 * j + 1]);
        }
      }
      wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < DH / 16; ++ks) mma_s_rs(dst, qr[ks], dsw(kslot + ks * 32), 1);
      wgmma_commit();
    };
    // softmax of block n (its QK retired by the caller's wait), then wait for PV(n-1), rescale, issue PV(n)
    auto soft_pv = [&](float* sc_, int n, int tt, bool more) {
#pragma unroll
      for (int j = 0; j < 32; ++j) fence_reg(sc_[j]);
      float alpha[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float mx = -INFINITY;
#pragma unroll
        for (int j = 0; j < 8; ++j) mx = fmaxf(mx, fmaxf(sc_[4 * j + 2 * h], sc_[4 * j + 2 * h + 1]));
        const float m_new = fmaxf(m_i[h], qmax(mx));
        alpha[h] = ex2(m_i[h] - m_new);
        m_i[h] = m_new;
        float ssum = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float p0 = ex2(sc_[4 * j + 2 * h] - m_new), p1 = ex2(sc_[4 * j + 2 * h + 1] - m_new);
          ssum += p0 + p1;
          const __nv_bfloat162 pk = __floats2bfloat162_rn(p0, p1);
          pa[j >> 1][2 * (j & 1) + h] = *reinterpret_cast<const uint32_t*>(&pk);
        }
        l_i[h] = l_i[h] * alpha[h] + ssum;
      }
      // PV(n-1) must have landed before acc is rescaled; QK(n+1) (committed after it) may stay in flight
      if (more) wgmma_wait<1>(); else wgmma_wait<0>();
#pragma unroll
      for (int j = 0; j < 24; ++j) fence_reg(acc[j]);
      if (n > 0 && lane == 0) mbar_arrive(&kempty[(tt - 1) % STAGES]);   // block n-1's K / V are no longer read
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int j = 0; j < 6; ++j) { acc[4 * j + 2 * h] *= alpha[h]; acc[4 * j + 2 * h + 1] *= alpha[h]; }
      wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < BN / 16; ++ks) mma_o(acc, pa[ks], dmn(sbase + OFF_KV + (tt % STAGES) * 2 * SKV + SKV, ks));
      wgmma_commit();
    };
    static_assert(NKB % 2 == 0, "the pipelined loop is unrolled by two");
    float sc1[32];
    seed_qk(sc, 0, t);
    for (int n = 0; n < NKB; n += 2, t += 2) {
      // groups in flight at the top: QK(n) [+ PV(n-1)]
      seed_qk(sc1, n + 1, t + 1);                                   // + QK(n+1)
      if (n == 0) wgmma_wait<1>(); else wgmma_wait<2>();          // QK(n) retired (PV(n-1), QK(n+1) may pend)
      soft_pv(sc, n, t, true);                                      // waits PV(n-1) (QK(n+1) pends), issues PV(n)
      const bool last = n + 2 >= NKB;
      if (!last) seed_qk(sc, n + 2, t + 2);                         // + QK(n+2)
      if (last) wgmma_wait<1>(); else wgmma_wait<2>();              // QK(n+1) retired
      soft_pv(sc1, n + 1, t + 1, !last);
    }
    wgmma_wait<0>();
    if (lane == 0) mbar_arrive(&kempty[(t - 1) % STAGES]);
#pragma unroll
    for (int j = 0; j < 24; ++j) fence_reg(acc[j]);
#else
    for (int n = 0; n < NKB; ++n, ++t) {
      const int s = t % STAGES;
      const uint32_t kslot = sbase + OFF_KV + s * 2 * SKV;
      mbar_wait(&kfull[s], (t / STAGES) & 1);
      if (!FLOOR) {
        float mk[16];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if (HASM) { const float2 v = *reinterpret_cast<const float2*>(kmr + n * BN + j * 8 + cb); mk[2 * j] = v.x; mk[2 * j + 1] = v.y; }
          else { mk[2 * j] = 0.f; mk[2 * j + 1] = 0.f; }
        }
        const uint32_t sbn = sbase + OFF_B + n * SB + wg * 8192;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          uint32_t bm[2][4];
#pragma unroll
          for (int half = 0; half < 2; ++half)
            ldsm_x4(bm[half], sbn + sw128(16 * warp + 8 * h + (lane & 7), 8 * (4 * half + (lane >> 3)) * 2));
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const float2 b = bf2f(bm[j >> 2][j & 3]);
            sc[4 * j + 2 * h] = b.x + (mk[2 * j]);
            sc[4 * j + 2 * h + 1] = b.y + (mk[2 * j + 1]);
          }
        }
        wgmma_fence();
#pragma unroll
        for (int ks = 0; ks < DH / 16; ++ks) mma_s_rs(sc, qr[ks], dsw(kslot + ks * 32), 1);
        wgmma_commit();
        wgmma_wait<0>();
#pragma unroll
        for (int j = 0; j < 32; ++j) fence_reg(sc[j]);
        float alpha[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          float mx = -INFINITY;
#pragma unroll
          for (int j = 0; j < 8; ++j) mx = fmaxf(mx, fmaxf(sc[4 * j + 2 * h], sc[4 * j + 2 * h + 1]));
          const float m_new = fmaxf(m_i[h], qmax(mx));
          alpha[h] = ex2(m_i[h] - m_new);
          m_i[h] = m_new;
          float ssum = 0.f;
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const float p0 = ex2(sc[4 * j + 2 * h] - m_new), p1 = ex2(sc[4 * j + 2 * h + 1] - m_new);
            ssum += p0 + p1;
            const __nv_bfloat162 pk = __floats2bfloat162_rn(p0, p1);
            pa[j >> 1][2 * (j & 1) + h] = *reinterpret_cast<const uint32_t*>(&pk);
          }
          l_i[h] = l_i[h] * alpha[h] + ssum;
        }
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int j = 0; j < 6; ++j) { acc[4 * j + 2 * h] *= alpha[h]; acc[4 * j + 2 * h + 1] *= alpha[h]; }
        wgmma_fence();
#pragma unroll
        for (int j = 0; j < 24; ++j) fence_reg(acc[j]);
#pragma unroll
        for (int ks = 0; ks < BN / 16; ++ks) mma_o(acc, pa[ks], dmn(kslot + SKV, ks));
        wgmma_commit();
        wgmma_wait<0>();
      } else {
        acc[0] += __int_as_float(*reinterpret_cast<const int*>(sm + OFF_KV + s * 2 * SKV + lane * 4));
        m_i[0] = m_i[1] = 0.f; l_i[0] = l_i[1] = 1.f;
      }
      if (lane == 0) mbar_arrive(&kempty[s]);
    }
#pragma unroll
    for (int j = 0; j < 24; ++j) fence_reg(acc[j]);
#endif
    l_i[0] = qsum(l_i[0]); l_i[1] = qsum(l_i[1]);

    const int xb = i & 1;
    if (!owner) {                                                   // send (acc, m, l) to the partner, which owns the rows
      mbar_wait(&xempty[xb], ((i >> 1) & 1) ^ 1);
      const uint32_t prt = rank ^ 1;
      const uint32_t dst = mapa(sbase + OFF_X + xb * SX, prt), cbar = mapa(smem_u32(&xfull[xb]), prt);
      float v[XFL];
#pragma unroll
      for (int j = 0; j < 24; ++j) v[j] = acc[j];
      v[24] = m_i[0]; v[25] = m_i[1]; v[26] = l_i[0]; v[27] = l_i[1];
#pragma unroll
      for (int k = 0; k < XFL / 4; ++k) st_async4(dst + (k * 128 + wt) * 16, v[4 * k], v[4 * k + 1], v[4 * k + 2], v[4 * k + 3], cbar);
      continue;
    }
    if (KS > 1) {                                                   // combine with the partner's half of the keys
      if (wt == 0) mbar_arrive_expect_tx(&xfull[xb], SX);
      mbar_wait(&xfull[xb], (i >> 1) & 1);
      float v[XFL];
      const float4* xs = reinterpret_cast<const float4*>(sm + OFF_X + xb * SX);
#pragma unroll
      for (int k = 0; k < XFL / 4; ++k) {
        const float4 f = xs[k * 128 + wt];
        v[4 * k] = f.x; v[4 * k + 1] = f.y; v[4 * k + 2] = f.z; v[4 * k + 3] = f.w;
      }
      __syncwarp();
      if (lane == 0) mbar_arrive_remote(&xempty[xb], rank ^ 1);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float mm = fmaxf(m_i[h], v[24 + h]);
        const float s0 = ex2(m_i[h] - mm), s1 = ex2(v[24 + h] - mm);
#pragma unroll
        for (int j = 0; j < 6; ++j) {
          acc[4 * j + 2 * h] = acc[4 * j + 2 * h] * s0 + v[4 * j + 2 * h] * s1;
          acc[4 * j + 2 * h + 1] = acc[4 * j + 2 * h + 1] * s0 + v[4 * j + 2 * h + 1] * s1;
        }
        l_i[h] = l_i[h] * s0 + v[26 + h] * s1;
        m_i[h] = mm;
      }
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float inv = 1.f / l_i[h];
      const int row = m0 + wg * 64 + r0 + 8 * h;
      float* orow = O + ((size_t)a * L + row) * DM + qcol;
#pragma unroll
      for (int j = 0; j < 6; ++j)
        *reinterpret_cast<float2*>(orow + j * 8 + cb) = make_float2(acc[4 * j + 2 * h] * inv, acc[4 * j + 2 * h + 1] * inv);
      if ((lane & 3) == 0) LSE[((size_t)a * H + head) * L + row] = m_i[h] + __log2f(l_i[h]);
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
void launch(const CUtensorMap& mq, const CUtensorMap& mk, const CUtensorMap& mv, const CUtensorMap& mb, const float* km,
            float* o, float* lse, int L, int H, int A, int sch) {
  const size_t smem = 1024 + OFF_BAR + 256;
  auto kern = attn_fwd2_kernel<HASM, KS>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  const int mt = L / QM;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(KS * mt * (A / sch) * H);
  cfg.blockDim = dim3(128 * NWG + 32);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = KS; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, kern, mq, mk, mv, mb, km, o, lse, L, H, mt, sch) == cudaSuccess, "launch failed");
}
}  // namespace

// as attn_fwd.cu, plus sch: samples per CTA (divides A)
void attn_fwd2(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bias, c10::optional<torch::Tensor> kmask,
               torch::Tensor O, torch::Tensor LSE, int64_t sch) {
  const int H = bias.size(0), L = bias.size(1);
  const int A = q.size(0) / L;
  for (auto* t : {&q, &k, &v})
    TORCH_CHECK(t->is_contiguous() && t->scalar_type() == torch::kBFloat16 && t->size(1) == DM && t->size(0) == A * L, "qkv layout");
  TORCH_CHECK(bias.is_contiguous() && bias.scalar_type() == torch::kBFloat16 && bias.size(2) == L && H * DH == DM, "bias layout");
  TORCH_CHECK(O.is_contiguous() && O.scalar_type() == torch::kFloat32 && LSE.is_contiguous(), "out layout");
  TORCH_CHECK(L % QM == 0 && L % KR == 0 && (L / KR == 1 || L / KR == 2) && A % sch == 0, "shape");
  const bool hasm = kmask.has_value() && kmask->defined();
  auto mq = tile_map(q.data_ptr(), (uint64_t)A * L, DM, DM, QW, QM);
  auto mk = tile_map(k.data_ptr(), (uint64_t)A * L, DM, DM, QW, BN);
  auto mv = tile_map(v.data_ptr(), (uint64_t)A * L, DM, DM, QW, BN);
  auto mb = tile_map(bias.data_ptr(), (uint64_t)H * L, L, L, BN, QM);
  const float* km = hasm ? kmask->data_ptr<float>() : nullptr;
  float *o = O.data_ptr<float>(), *lse = LSE.data_ptr<float>();
  const int ks = L / KR;
  if (hasm) { if (ks == 1) launch<true, 1>(mq, mk, mv, mb, km, o, lse, L, H, A, sch); else launch<true, 2>(mq, mk, mv, mb, km, o, lse, L, H, A, sch); }
  else { if (ks == 1) launch<false, 1>(mq, mk, mv, mb, km, o, lse, L, H, A, sch); else launch<false, 2>(mq, mk, mv, mb, km, o, lse, L, H, A, sch); }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("attn_fwd2", &attn_fwd2); }
