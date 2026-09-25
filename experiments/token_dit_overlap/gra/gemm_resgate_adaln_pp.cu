// gemm_resgate_adaln_pp.cu -- the residual GEMM with the gated residual and the next AdaLN in its epilogue, persistent
// and ping-ponged (sm_90a). Same contract as gemm_resgate_adaln.cu (v4):
//
//   acc = A W^T                    A [M,K] bf16 (row stride sa), W [768,K] bf16
//   x  += sigmoid(gl[tok]) * acc   x [M,768] fp32, in place
//   xa  = LN(x) * sigmoid(ms[tok]) + mb[tok]        bf16, only when ADALN
//
// Why v4 lost: one tile per CTA, one wave, so every CTA ran its epilogue at the same moment, after its mainloop -- the
// ~30 MB of x read-modify-write + xa write was added to the tensor time instead of hidden under it. Here each cluster
// walks a list of 64-row tiles and its two consumer warpgroups take alternate tiles: while one runs tile i's epilogue
// (memory), the other runs tile i+1's mainloop (tensor cores). The ring is filled in tile order, so the mainloops
// serialise through it by themselves.
//
// Cluster: 4 CTAs along N (192 columns each); A is multicast (each CTA loads 16 of the tile's 64 rows). Per CTA: one
// producer warpgroup (one issuing thread), two consumer warpgroups. Shared memory: an ST-stage ring of A [64 x 64] +
// W [192 x 64], and one x tile [64 x 192] fp32 per consumer warpgroup (TMA in, updated in place, TMA out, then read back
// for the AdaLN). gl, ms and mb are read straight from global (they are [L, *], L2-resident) and xa goes out with
// 16-byte stores, which leaves the ring room for three stages.
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;

#ifndef EPIREG
#define EPIREG 1                 // 1: the epilogue works from registers -- x in and out and xa out straight from the
#endif                           // wgmma fragment -- so the two 48 KB x tiles are gone and the ring gets six stages.
                                 // The 3-stage ring was TMA-latency-bound: ~0.65 us a 32 KB chunk against 0.21 of MMA,
                                 // and halving W's L2 traffic (CM = 2) did not move it. 0: x staged in shared memory.
#ifndef STAGES
#define STAGES (EPIREG ? 6 : 3)
#endif
#ifndef PDL
#define PDL 1
#endif
#ifdef PP_TIMES
// [CTA][local tile][stamp], %globaltimer ns: 0 before the turn wait, 1 mainloop start, 2 last chunk waited, 3 acc
// ready, 4 x tile landed, 5 residual done, 6 stats landed, 7 xa written, 8 x tile freed. [CTA][15][0] = kernel start.
__device__ unsigned long long g_pp[512][16][10];
TMN_DEVI unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define PST(i, k) do { if (ti == 0 && (i) < 15) g_pp[blockIdx.x][(i)][(k)] = gtimer(); } while (0)
#else
#define PST(i, k) do { } while (0)
#endif

#ifndef CM
#define CM 2                     // CTAs along M in a cluster: 2 pairs adjacent 64-row tiles and multicasts W between
#endif                           // them (each loads half), halving W's L2 traffic; 1 is the 4-CTA layout
constexpr int D_ = 768, NC = 192, CL = 4, KC = 64, TM = 64;
constexpr int CLT = CL * CM;                                        // cluster size: rank = n-rank + CL * m-rank
constexpr int XB = NC / 32, TILE = 8192;                            // x tile: 6 boxes of [64 rows][32 fp32]
constexpr int SA = TM * 128, SB = NC * 128, SS = SA + SB;           // ring stage: A 8 KB + W 24 KB
constexpr int OX = STAGES * SS, XT = XB * TILE;                     // two x tiles after the ring
constexpr int OST = OX + (EPIREG ? 0 : 2 * XT);                    // stats [wg][parity][CL src][TM] (mean, M2)
constexpr int OMR = OST + 2 * 2 * CL * TM * 8;                      // merged [wg][TM] (mean, rstd)
constexpr int OBAR = OMR + 2 * TM * 8;
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
TMN_DEVI void tma_store_2d(const CUtensorMap* map, const void* src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
               :: "l"(map), "r"(smem_u32(src)), "r"(c0), "r"(c1) : "memory");
}
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
TMN_DEVI uint32_t sw128(int row, int byte) { return row * 128 + ((((byte >> 4) ^ (row & 7))) << 4) + (byte & 15); }

template <bool ADALN>
// 384 threads, and the producer warpgroup hands its registers to the consumers (setmaxnreg 40 / 232). At the launch
// cap of 168 the AdaLN variant spilled 576 bytes of stack (its ms / mb prefetch), which took the AdaLN phase to
// 10-13 us a tile. A 288-thread block (one producer warp) does not help: ptxas still budgets 168.
__global__ void __launch_bounds__(384, 1)
grapp_kernel(const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mw,
             const __grid_constant__ CUtensorMap mx, const __nv_bfloat16* __restrict__ GL,
             const __nv_bfloat16* __restrict__ MS, const __nv_bfloat16* __restrict__ MB,
             __nv_bfloat16* __restrict__ XA, float* __restrict__ X, int L, int K, int ntiles, int sgl, int sms,
             int smb, float eps) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + OBAR);
  uint64_t* empty = full + STAGES;
  uint64_t* xfull = empty + STAGES;                                 // [wg]: this warpgroup's x tile has landed
  uint64_t* xfree = xfull + 2;                                      // [wg]: its x tile is free for the next load
  uint64_t* sbar = xfree + 2;                                       // [wg][parity]: the four CTAs' row stats landed
  // [wg]: the other warpgroup has waited on every chunk of its tile, so this one may start its mainloop. Without it the
  // warpgroup taking tile i + 1 waited on its first chunk's full barrier while the other was still several phases
  // behind on it, and a parity wait then passes on an OLDER phase with the same parity (ABA): it read a stale slot,
  // released it early, peers multicast into a slot still in use, and the tx accounting broke (the producer faulted on
  // its next arrive.expect_tx). A waiter must never be more than one phase ahead of the barrier.
  uint64_t* turn = sbar + 4;
  float2* stats = reinterpret_cast<float2*>(sm + OST);
  float2* mrs = reinterpret_cast<float2*>(sm + OMR);

  const int tid = threadIdx.x, wg = tid >> 7;                       // wg 0, 1: consumers; 2: producer
  const uint32_t rank = cluster_rank(), nr = rank % CL, mr = rank / CL;
  const int cid = cluster_id_x(), ncl = nclusters_x();
  const int n0 = nr * NC, nk = K / KC, ngroups = ntiles / CM;     // a group: CM adjacent tiles, one per m-rank
  const int nloc = cid < ngroups ? (ngroups - cid + ncl - 1) / ncl : 0;   // tiles this CTA walks
  const int total = nloc * nk;                                      // ring chunks this CTA sees

  if (tid == 0) {
    // a slot is written by its A peers (same m) and W peers (same n); every consumer warp of the cluster frees it
    for (int s = 0; s < STAGES; ++s) { mbar_init(&full[s], 1); mbar_init(&empty[s], CLT * 4); }
    for (int w = 0; w < 2; ++w) { mbar_init(&xfull[w], 1); mbar_init(&xfree[w], 1); }
    for (int i = 0; i < 4; ++i) mbar_init(&sbar[i], 1);
    for (int w = 0; w < 2; ++w) mbar_init(&turn[w], 1);
    fence_barrier_init();
  }
  __syncthreads();
  if (ADALN && tid == 0)                                            // armed for the first use of each (wg, parity)
    for (int i = 0; i < 4; ++i) mbar_arrive_expect_tx(&sbar[i], STAT_BYTES);
  cluster_arrive(); cluster_wait();                                 // peers' barriers exist before anyone multicasts
#if PDL
  asm volatile("griddepcontrol.wait;" ::: "memory");                // A is the previous kernel's output
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // they still wait for our completion
#endif

  if (wg == 2) setmaxnreg_dec<40>(); else setmaxnreg_inc<232>();    // 128 x 40 + 256 x 232 = 64512 = 384 x 168
  if (wg == 2) {                                                    // ---- producer: one thread feeds everything
    if (tid == 256) {
      tma_prefetch_desc(&ma); tma_prefetch_desc(&mw); tma_prefetch_desc(&mx);
      for (int i = 0; i < nloc; ++i) {
        const int m0 = ((cid + i * ncl) * CM + mr) * TM, w = i & 1;
        const uint32_t xph = ((i >> 1) & 1) ^ 1;                    // k-th use of this warpgroup's x tile
        bool xdone = EPIREG;                                        // the register epilogue loads x itself
        auto issue_x = [&]() {
          mbar_arrive_expect_tx(&xfull[w], XT);
          for (int b = 0; b < XB; ++b) tma_load_2d(sm + OX + w * XT + b * TILE, &mx, &xfull[w], n0 + 32 * b, m0);
          xdone = true;
        };
        for (int kc = 0; kc < nk; ++kc) {
          const int c = i * nk + kc, s = c % STAGES;
          mbar_wait(&empty[s], ((c / STAGES) & 1) ^ 1);
          mbar_arrive_expect_tx(&full[s], SS);
          // A: this tile's rows, a quarter from each of the CL CTAs that share them (same m-rank)
          tma_load_2d_mc(sm + s * SS + nr * (SA / CL), &ma, &full[s], kc * KC, m0 + nr * (TM / CL),
                         (uint16_t)(((1u << CL) - 1) << (mr * CL)));
          // W: this n-rank's 192 columns, a 1/CM share from each of the CM CTAs that share them (same n-rank)
          if (CM == 1) tma_load_2d(sm + s * SS + SA, &mw, &full[s], kc * KC, n0);
          else tma_load_2d_mc(sm + s * SS + SA + mr * (SB / CM), &mw, &full[s], kc * KC, n0 + mr * (NC / CM),
                              (uint16_t)((1u << nr) | (1u << (nr + CL))));
          // the x tile goes in as soon as its warpgroup has let go of the previous one, without holding up the ring
          if (!xdone && mbar_try_wait(&xfree[w], xph)) issue_x();
        }
        if (!EPIREG && !xdone) { mbar_wait(&xfree[w], xph); issue_x(); }
      }
    }
    __syncwarp();
    cluster_arrive_relaxed(); cluster_wait();
    return;
  }

  // ---- consumers: warpgroup wg takes this cluster's local tiles wg, wg + 2, ...
  const int lane = tid & 31, warp = (tid >> 5) & 3, ti = tid & 127;
#ifdef PP_TIMES
  if (tid == 0) g_pp[blockIdx.x][15][0] = gtimer();
#endif
  const int rr0 = warp * 16 + (lane >> 2), cb = 2 * (lane & 3);
  uint8_t* smx = sm + OX + wg * XT;
  const uint32_t sbase = smem_u32(sm);
  constexpr int NCH = TM * (NC / 8) / 128;                          // AdaLN: 12 chunks of 8 columns a thread
  for (int i = wg, j = 0; i < nloc; i += 2, ++j) {                  // j: this warpgroup's use count
    const int m0 = ((cid + i * ncl) * CM + mr) * TM, t0 = m0 % L;
    float acc[3][32];
    PST(i, 0);
    if (i > 0) mbar_wait(&turn[wg], ((i - 1) >> 1) & 1);           // the other warpgroup is through tile i - 1's chunks
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
      // release the previous chunk; a release nobody will wait for (the last STAGES chunks) is not issued, or it could
      // still be in flight when the CTA exits (see the attention core's NODANGLE)
      if (kc > 0 && c - 1 + STAGES < total && lane == 0)
#pragma unroll
        for (int k = 0; k < CLT; ++k) mbar_arrive_remote(&empty[(c - 1) % STAGES], k);
    }
    named_bar_sync(1 + wg, 128);                                    // every thread has waited on every chunk of this tile
    if (ti == 0 && i + 1 < nloc) mbar_arrive(&turn[wg ^ 1]);
    PST(i, 2);
    wgmma_wait<0>();
    PST(i, 3);
    {
      const int c = i * nk + nk - 1;
      if (c + STAGES < total && lane == 0)
#pragma unroll
        for (int k = 0; k < CLT; ++k) mbar_arrive_remote(&empty[c % STAGES], k);
    }
#pragma unroll
    for (int q = 0; q < 3; ++q) fence_regs(acc[q]);

    // ---- epilogue. Fragment of m64nN: warp w, lane l holds rows 16w + l/4 (+8), columns 8q + 2(l%4) + {0,1}.
#if EPIREG
    {
      // x and gl one 64-column block ahead: a quad of lanes covers 32 contiguous bytes of an x row (a full sector)
      const size_t ro[2] = {(size_t)(m0 + rr0) * D_ + n0 + cb, (size_t)(m0 + rr0 + 8) * D_ + n0 + cb};
      const __nv_bfloat16* gr[2] = {GL + (size_t)(t0 + rr0) * sgl + n0 + cb, GL + (size_t)(t0 + rr0 + 8) * sgl + n0 + cb};
      float2 xv[3][2][8]; uint32_t gv[3][2][8];
      auto ld = [&](int q) {
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            xv[q][h][e] = *reinterpret_cast<const float2*>(X + ro[h] + q * 64 + e * 8);   // written here: not .nc
            gv[q][h][e] = __ldg(reinterpret_cast<const unsigned int*>(gr[h] + q * 64 + e * 8));
          }
      };
      ld(0);
      PST(i, 4);
      float sum[2] = {0.f, 0.f};
#pragma unroll
      for (int q = 0; q < 3; ++q) {
        if (q + 1 < 3) ld(q + 1);
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const float2 g = bf2f(gv[q][h][e]);
            float& e0 = acc[q][4 * e + 2 * h];
            float& e1 = acc[q][4 * e + 2 * h + 1];
            e0 = xv[q][h][e].x + sigmoid_t(g.x) * e0;
            e1 = xv[q][h][e].y + sigmoid_t(g.y) * e1;
            sum[h] += e0 + e1;
            *reinterpret_cast<float2*>(X + ro[h] + q * 64 + e * 8) = make_float2(e0, e1);
          }
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
          for (int q = 0; q < 3; ++q)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              const float d0 = acc[q][4 * e + 2 * h] - mean[h], d1 = acc[q][4 * e + 2 * h + 1] - mean[h];
              m2[h] += d0 * d0 + d1 * d1;
            }
          m2[h] += __shfl_xor_sync(0xffffffffu, m2[h], 1);
          m2[h] += __shfl_xor_sync(0xffffffffu, m2[h], 2);
        }
        if ((lane & 3) == 0)
#pragma unroll
          for (int k = 0; k < CL; ++k) {                            // the CL CTAs that own these rows
            st_async_f2(&st[nr * TM + rr0], sb, mr * CL + k, mean[0], m2[0]);
            st_async_f2(&st[nr * TM + rr0 + 8], sb, mr * CL + k, mean[1], m2[1]);
          }
        // ms / mb in the fragment's own layout, under the peers' statistics
        const __nv_bfloat16* sr[2] = {MS + (size_t)(t0 + rr0) * sms + n0 + cb, MS + (size_t)(t0 + rr0 + 8) * sms + n0 + cb};
        const __nv_bfloat16* br[2] = {MB + (size_t)(t0 + rr0) * smb + n0 + cb, MB + (size_t)(t0 + rr0 + 8) * smb + n0 + cb};
        uint32_t sv[3][2][8], bv[3][2][8];
#pragma unroll
        for (int q = 0; q < 3; ++q)
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              sv[q][h][e] = __ldg(reinterpret_cast<const unsigned int*>(sr[h] + q * 64 + e * 8));
              bv[q][h][e] = __ldg(reinterpret_cast<const unsigned int*>(br[h] + q * 64 + e * 8));
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
        if (ti == 0) mbar_arrive_expect_tx(sb, STAT_BYTES);       // re-armed for this (wg, parity)'s next use
        const float2 mv[2] = {mrs[wg * TM + rr0], mrs[wg * TM + rr0 + 8]};
        const size_t xo[2] = {(size_t)(m0 + rr0) * D_ + n0 + cb, (size_t)(m0 + rr0 + 8) * D_ + n0 + cb};
#pragma unroll
        for (int q = 0; q < 3; ++q)
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              const float2 sc = bf2f(sv[q][h][e]), sh = bf2f(bv[q][h][e]);
              const float o0 = (acc[q][4 * e + 2 * h] - mv[h].x) * mv[h].y * sigmoid_t(sc.x) + sh.x;
              const float o1 = (acc[q][4 * e + 2 * h + 1] - mv[h].x) * mv[h].y * sigmoid_t(sc.y) + sh.y;
              *reinterpret_cast<__nv_bfloat162*>(XA + xo[h] + q * 64 + e * 8) = __floats2bfloat162_rn(o0, o1);
            }
        // mrs is read again for this warpgroup's next tile: nobody may overwrite it before everyone has read it
        named_bar_sync(1 + wg, 128);
      }
      PST(i, 7);
    }
#else
    // gl does not depend on anything this tile writes: fetch all of it now, under the x tile's landing, instead of one
    // round trip per 64-column block inside the residual loop
    uint32_t gv[3][2][8];
#pragma unroll
    for (int q = 0; q < 3; ++q)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const __nv_bfloat16* grow = GL + (size_t)(t0 + rr0 + 8 * h) * sgl + n0;
#pragma unroll
        for (int e = 0; e < 8; ++e) gv[q][h][e] = __ldg(reinterpret_cast<const unsigned int*>(grow + q * 64 + e * 8 + cb));
      }
    mbar_wait(&xfull[wg], j & 1);
    PST(i, 4);
    float sum[2] = {0.f, 0.f};
#pragma unroll
    for (int q = 0; q < 3; ++q) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int rr = rr0 + 8 * h;
        float2 xv[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int col = q * 64 + e * 8 + cb;
          xv[e] = *reinterpret_cast<const float2*>(smx + (col >> 5) * TILE + sw128(rr, (col & 31) * 4));
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float2 g = bf2f(gv[q][h][e]);
          float& e0 = acc[q][4 * e + 2 * h];
          float& e1 = acc[q][4 * e + 2 * h + 1];
          e0 = xv[e].x + sigmoid_t(g.x) * e0;
          e1 = xv[e].y + sigmoid_t(g.y) * e1;
          sum[h] += e0 + e1;
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int col = q * 64 + e * 8 + cb;
          *reinterpret_cast<float2*>(smx + (col >> 5) * TILE + sw128(rr, (col & 31) * 4)) =
              make_float2(acc[q][4 * e + 2 * h], acc[q][4 * e + 2 * h + 1]);
        }
      }
      fence_proxy_async();
      named_bar_sync(1 + wg, 128);
      if (ti == 0) {
        for (int b = 2 * q; b < 2 * q + 2; ++b) tma_store_2d(&mx, smx + b * TILE, n0 + 32 * b, m0);
        tma_store_commit();
      }
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
        for (int q = 0; q < 3; ++q)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const float d0 = acc[q][4 * e + 2 * h] - mean[h], d1 = acc[q][4 * e + 2 * h + 1] - mean[h];
            m2[h] += d0 * d0 + d1 * d1;
          }
        m2[h] += __shfl_xor_sync(0xffffffffu, m2[h], 1);
        m2[h] += __shfl_xor_sync(0xffffffffu, m2[h], 2);
      }
      if ((lane & 3) == 0)
#pragma unroll
        for (int k = 0; k < CL; ++k) {                              // the CL CTAs that own these rows
          st_async_f2(&st[nr * TM + rr0], sb, mr * CL + k, mean[0], m2[0]);
          st_async_f2(&st[nr * TM + rr0 + 8], sb, mr * CL + k, mean[1], m2[1]);
        }
      uint4 msv[NCH], mbv[NCH];
#pragma unroll
      for (int u = 0; u < NCH; ++u) {
        const int c = ti + 128 * u, r = c / 24, col = 8 * (c % 24);
        const size_t tok = t0 + r;
        msv[u] = __ldg(reinterpret_cast<const uint4*>(MS + tok * sms + n0 + col));
        mbv[u] = __ldg(reinterpret_cast<const uint4*>(MB + tok * smb + n0 + col));
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
#pragma unroll
      for (int u = 0; u < NCH; ++u) {
        const int c = ti + 128 * u, r = c / 24, col = 8 * (c % 24);
        const float2 mr = mrs[wg * TM + r];
        const int g = (col & 31) >> 2;
        const uint8_t* xrow = smx + (col >> 5) * TILE + r * 128;
        const float4 x0 = *reinterpret_cast<const float4*>(xrow + ((g ^ (r & 7)) << 4));
        const float4 x1 = *reinterpret_cast<const float4*>(xrow + (((g + 1) ^ (r & 7)) << 4));
        const float xs[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        const uint32_t sw[4] = {msv[u].x, msv[u].y, msv[u].z, msv[u].w}, bw[4] = {mbv[u].x, mbv[u].y, mbv[u].z, mbv[u].w};
        uint32_t o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 sc = bf2f(sw[e]), sh = bf2f(bw[e]);
          const float o0 = (xs[2 * e] - mr.x) * mr.y * sigmoid_t(sc.x) + sh.x;
          const float o1 = (xs[2 * e + 1] - mr.x) * mr.y * sigmoid_t(sc.y) + sh.y;
          __nv_bfloat162 v = __floats2bfloat162_rn(o0, o1);
          o[e] = *reinterpret_cast<uint32_t*>(&v);
        }
        *reinterpret_cast<uint4*>(XA + (size_t)(m0 + r) * D_ + n0 + col) = make_uint4(o[0], o[1], o[2], o[3]);
      }
    }
    PST(i, 7);
    // the x tile is free once the TMA stores have read it and every thread is done reading it back
    if (ti == 0) tma_store_wait_read<0>();
    named_bar_sync(1 + wg, 128);
    if (ti == 0) mbar_arrive(&xfree[wg]);
#endif
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

template <bool ADALN>
void launch(const torch::Tensor& a, const torch::Tensor& w, torch::Tensor& x, const torch::Tensor& gl,
            const torch::Tensor* ms, const torch::Tensor* mb, torch::Tensor* xa, int64_t L, double eps, int64_t max_cl) {
  const int M = a.size(0), K = a.size(1), ntiles = M / TM, ngroups = ntiles / CM;
  TORCH_CHECK(K % KC == 0 && M % (TM * CM) == 0 && L % TM == 0 && M % L == 0, "shape: K % 64, M % (64 CM), L % 64");
  auto kern = grapp_kernel<ADALN>;
  static int maxc = [&] {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    cudaLaunchConfig_t q{};
    q.gridDim = dim3(CLT * 16); q.blockDim = dim3(384); q.dynamicSmemBytes = SMEM_BYTES;
    cudaLaunchAttribute a1[1];
    a1[0].id = cudaLaunchAttributeClusterDimension;
    a1[0].val.clusterDim.x = CLT; a1[0].val.clusterDim.y = 1; a1[0].val.clusterDim.z = 1;
    q.attrs = a1; q.numAttrs = 1;
    int n = 0;
    TORCH_CHECK(cudaOccupancyMaxActiveClusters(&n, kern, &q) == cudaSuccess && n > 0, "no cluster fits");
    return n;
  }();
  int ncl = std::min(ngroups, maxc);
  if (max_cl > 0) ncl = std::min<int>(ncl, max_cl);
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(CLT * ncl);
  cfg.blockDim = dim3(384);
  cfg.dynamicSmemBytes = SMEM_BYTES;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = CLT; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[1].val.programmaticStreamSerializationAllowed = PDL;
  cfg.attrs = at; cfg.numAttrs = 2;
  auto bp = [](const torch::Tensor* t) { return t ? reinterpret_cast<const __nv_bfloat16*>(t->data_ptr()) : nullptr; };
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, kern, tile_map(a, 64, TM / CL), tile_map(w, 64, NC / CM), tile_map(x, 32, TM),
                                 reinterpret_cast<const __nv_bfloat16*>(gl.data_ptr()), bp(ms), bp(mb),
                                 xa ? reinterpret_cast<__nv_bfloat16*>(xa->data_ptr()) : nullptr,
                                 reinterpret_cast<float*>(x.data_ptr()), (int)L, K, ntiles, (int)gl.stride(0), ms ? (int)ms->stride(0) : 0,
                                 mb ? (int)mb->stride(0) : 0, (float)eps) == cudaSuccess, "launch failed");
}
}  // namespace

// x += sigmoid(gl) * (a @ w^T); then, when ms is given, xa = AdaLN(x). max_clusters > 0 caps the persistent grid.
void gemm_resgate_adaln_pp(torch::Tensor a, torch::Tensor w, torch::Tensor x, torch::Tensor gl,
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
  m.def("gemm_resgate_adaln_pp", &gemm_resgate_adaln_pp);
  m.def("debug_times", &debug_times);
}
