// attn_core.cu -- the token DiT attention core as one sm_90a kernel: gated attention over a packed q|k|v|g buffer with
// the hoisted pair bias, for S samples that share that bias.
//
//   S[i] = q[i] k[i]^T + bias          bias is per (block, head), the SAME tensor for every sample
//   o[i] = softmax(S[i]) v[i]
//   out  = sigmoid(g[i]) * o[i]        written over q, as the Triton core does
//
// Why it exists: in the packaged Triton core each sample reads the whole bias, 5 x 18.9 MB from L2 per block at L768,
// which the ablation prices at 11.3 us of the core's 50.0. Here the S CTAs that share a bias tile are one cluster and the
// tile arrives by TMA multicast, so it is read once: ~19 MB. Triton cannot express this (a cluster with device-side
// descriptors fails to compile, and num_ctas must divide the grid, which S = 5 does not).
//
// Head dim 48 needs no padding: K = 48 is three wgmma k-steps of 16. q|k|v|g live in one [S*L, 4D] buffer, so a head's
// 48 columns are loaded 64 wide (the extra 16 belong to the next field and are never read by the three k-steps).
// P converts from the S accumulator to the PV A-operand by packing alone: for m64k16 the two layouts coincide.
#include "tmn_kernels.cuh"
using namespace tmn; using namespace tmn::sm90;

#ifndef STAGES
#define STAGES 3
#endif
#ifndef NWG
#define NWG 2                    // consumer warpgroups: query rows per CTA = 64 NWG
#endif
#ifndef MAXLESS
#define MAXLESS 1                // softmax without the running max: exact, and it drops the max reduction, the
#endif                           // rescale factor and the accumulator rescale. Safe while |logit| stays under ~120.
#ifndef RSPLIT
#define RSPLIT 1                   // 1: hand the producer's registers to the consumers (setmaxnreg)
#endif
#ifndef BLKSM
#define BLKSM 2                  // CTAs per SM: this kernel wants two resident (18 warps) and fits them at 96 registers
#endif
#ifndef PONG
#define PONG 0                   // staggering the consumer warpgroups once at the start bought nothing measurable
#endif
#ifndef PWARP
#define PWARP 1                  // 1: the producer is a single warp, not a whole warpgroup -- only one thread issues
#endif                           // TMA, and the other three warps only occupied scheduler slots
#ifndef SOFTPV
#define SOFTPV 0                 // interleaving the softmax with PV per k-step (the wgmma for the first 16 keys while
#endif                           // the next 16 are still in ex2) measured slower: 53.5 against 49.4
#ifndef DHOIST
#define DHOIST 0                 // one wgmma descriptor per slot up front costs more than it saves: the array spills
                                 // (48 bytes of stack) and the kernel goes 49.0 -> 52.2 us
#endif
#ifndef LSUM
#define LSUM 0                   // the softmax denominator from the tensor core: P times a tile of ones. It removes
#endif                           // 32 FADDs a block but the four extra wgmma cost more: 58.6 against 52.6 us
#ifndef MOFF
#define MOFF 0                   // subtract a per-row offset taken from the first key block. Mathematically a no-op
#endif                           // (softmax is shift-invariant) and it only buys exp2 headroom, but it costs 8 us a
                                 // block in the step whichever way it is written, and rel_rms moves 4.52e-3 -> 4.40e-3
#ifndef BACC
#define BACC 1                   // seed the score accumulator with the bias and let the QK wgmma accumulate onto it,
#endif                           // instead of adding the bias afterwards: 32 FADDs a key block a thread
#ifndef QREG
#define QREG 1                   // hold the q tile in registers (12 of them) instead of a 16 KB shared tile: it is read
#endif                           // once per key block as the wgmma A operand, and the space buys another ring stage
#ifndef NISS
#define NISS 3                   // TMA issuers per CTA (K, V, bias on their own threads). One issuing thread caps at
#endif                           // about 27 GB/s on an SM, and this loads 24 KB per key block per CTA
#ifndef BLDM
#define BLDM 1                   // read the bias tile with ldmatrix: one instruction per 4 column groups instead of
#endif                           // one 4-byte load per column pair (16 -> 4 per thread per key block)
#ifndef FLOOR
#define FLOOR 0                  // 1: the same TMA traffic and barriers, no math -- the measured pattern floor
#endif
#ifndef PIPE
#define PIPE 0                   // 1: software-pipelined (softmax of one key block under the previous block's PV)
#endif
constexpr int BN = 64, DH = 48, QW = 64, QM = 64 * NWG;    // key block, head dim, load width, query rows

TMN_DEVI uint64_t dsw(uint32_t addr) { return smem_desc(addr, 16, 1024, 1); }          // K-major, 128-B swizzle
TMN_DEVI uint64_t dmn(uint32_t base, int ks) { return smem_desc(base + ks * 2048, 16, 1024, 1); }   // MN-major, k-step = 16 rows
TMN_DEVI uint32_t sw128(int row, int byte) { return row * 128 + ((((byte >> 4) ^ (row & 7))) << 4) + (byte & 15); }
TMN_DEVI float sigmoid_t(float a) {
  float t; asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * a));
  return fmaf(0.5f, t, 0.5f);
}
TMN_DEVI float ex2(float a) { float r; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }

// S = q k^T: both operands K-major in shared memory
TMN_DEVI void mma_s(float* d, uint64_t a, uint64_t b, int accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %34, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 0, 0; }"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) : "l"(a), "l"(b), "r"(accumulate));
}
// S = q k^T with A (q) from registers: the same shape, one fewer shared-memory operand
TMN_DEVI void mma_s_rs(float* d, const uint32_t (&a)[4], uint64_t b, int accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %37, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 0; }"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b), "r"(accumulate));
}
// o += p v: A from registers, B (v, rows are keys) MN-major -> trans-b = 1
TMN_DEVI void mma_o(float (&d)[24], const uint32_t (&a)[4], uint64_t b) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23}, {%24,%25,%26,%27}, %28, p, 1, 1, 1; }"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b));
}
// l += p 1: the same A operand, B a tile of ones (MN-major); only the first of its 8 columns is read back
TMN_DEVI void mma_l(float (&d)[4], const uint32_t (&a)[4], uint64_t b) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, 1, 0; wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p, 1, 1, 1; }"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
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
TMN_DEVI void cluster_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory"); }
TMN_DEVI void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
TMN_DEVI uint32_t cluster_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r)); return r; }
// bar.arrive is the non-blocking half of a named barrier: the token's producer does not stall, the waiter does.
TMN_DEVI void named_bar_arrive(int id, int n) { __syncwarp(); asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory"); }
TMN_DEVI float2 bf2f(uint32_t u) { return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u)); }
TMN_DEVI float qmax(float v) {                                      // max over the four lanes of a quad = over the 64 keys
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}
TMN_DEVI float qsum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

// smem: q tile, then ST stages of (K, V, bias), then the barriers
constexpr int SQ = QREG ? 0 : QM * 128, SKV = BN * 128, SB = QM * 128;
constexpr int OFF_ST = SQ, ST_BYTES = 2 * SKV + SB;
constexpr int SONES = LSUM ? BN * 128 : 0;                          // a tile of ones for the denominator wgmma
constexpr int QSTAGE = 2 * SKV;                                     // where a QREG kernel parks the q tile: slot 0's bias area

template <int CLS>
__global__ void __launch_bounds__(128 * NWG + (PWARP ? 32 : 128), BLKSM)
attn_kernel(const __grid_constant__ CUtensorMap mq, const __grid_constant__ CUtensorMap mkv,
            const __grid_constant__ CUtensorMap mbias,
            __nv_bfloat16* __restrict__ OUT, int L, int H, int brow0, int mt, int S_, float* __restrict__ DBG) {
  (void)DBG;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint8_t* sones = sm + OFF_ST + STAGES * ST_BYTES;
  uint64_t* full = reinterpret_cast<uint64_t*>(sones + SONES);
  uint64_t* empty = full + STAGES;
  uint64_t* qbar = empty + STAGES;
  uint64_t* qdone = qbar + 1;                                       // the consumers have q in registers; slot 0 is free

  const int tid = threadIdx.x;
  const uint32_t rank = cluster_rank();                             // = the sample this CTA owns
  // The producer only issues TMA; give its registers to the consumers, which hold the score tile and the accumulator.
  // Register split, the way tmn_kernels.cuh does it: the producer keeps 40 and the consumers take the rest of the
  // CTA's launch allocation, 65536 / (threads * blocks per SM) rounded down to an 8-register granule, capped at 232.
  constexpr int NTHR = 128 * (NWG + 1), LAUNCH_REGS = (65536 / (NTHR * (NWG == 1 ? 2 : 1))) / 8 * 8;
  constexpr int CONS_REGS = (NTHR * LAUNCH_REGS - 128 * 40) / (128 * NWG) / 8 * 8 > 232
                            ? 232 : (NTHR * LAUNCH_REGS - 128 * 40) / (128 * NWG) / 8 * 8;
  if (RSPLIT) { if (tid >= 128 * NWG) setmaxnreg_dec<40>(); else setmaxnreg_inc<CONS_REGS>(); }
  // With a cluster the sample is the CTA's rank in it, so the S CTAs that share a bias tile are co-scheduled and the
  // tile can be multicast. CLS = 1 turns that off: the sample becomes the slowest grid dimension and each CTA loads
  // its own bias (the multicast with mask 1 is an ordinary load).
  const int ntile = mt * H;
  // Without a cluster the sample is the FASTEST grid dimension, so the S CTAs that share a bias tile are launched
  // together and all but the first read it out of L2. As the slowest dimension they ran a whole grid apart and every
  // one of them paid for the tile again.
  const int cid = CLS > 1 ? blockIdx.x / CLS : blockIdx.x / S_;
  const uint32_t samp = CLS > 1 ? rank : (uint32_t)(blockIdx.x % S_);
  const int m_tile = cid % mt, head = cid / mt;
  const int m0 = m_tile * QM, qcol = head * DH, row0 = samp * L + m0;
  const int wg = tid >> 7;
  const int nblocks = L / BN;

  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) { mbar_init(&full[s], NISS); mbar_init(&empty[s], CLS * 4 * NWG); }
    mbar_init(qbar, 1);
    mbar_init(qdone, 4 * NWG);
    fence_barrier_init();
  }
  __syncthreads();
  asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
  cluster_wait();

  if (tid >= 128 * NWG) {                                           // producer warp: one issuing thread per tensor
    const int iss = tid - 128 * NWG;                                // 0 = k, 1 = v, 2 = bias
    if (iss == 0) {
      tma_prefetch_desc(&mq); tma_prefetch_desc(&mkv); tma_prefetch_desc(&mbias);
      mbar_arrive_expect_tx(qbar, QM * 128);
      tma_load_2d(QREG ? sm + OFF_ST + QSTAGE : sm, &mq, qbar, qcol, row0);   // q tile: QM rows
    }
    if (QREG && iss < NISS) mbar_wait(qdone, 0);                    // q is out of slot 0 and in the consumers' registers
    if (iss < NISS) {
      for (int n = 0; n < nblocks; ++n) {
        const int s = n % STAGES;
        mbar_wait(&empty[s], ((n / STAGES) & 1) ^ 1);
        uint8_t* slot = sm + OFF_ST + s * ST_BYTES;
        if (NISS == 1) {                                            // one thread issues all three
          mbar_arrive_expect_tx(&full[s], ST_BYTES);
          tma_load_2d(slot, &mkv, &full[s], 768 + qcol, samp * L + n * BN);                        // k
          tma_load_2d(slot + SKV, &mkv, &full[s], 1536 + qcol, samp * L + n * BN);                 // v
          if (rank == (uint32_t)(n % CLS))                          // rotate the issuer: one CTA issuing every block's
            tma_load_mc(slot + 2 * SKV, &mbias, &full[s], n * BN, brow0 + head * L + m0, (1u << CLS) - 1);
        } else if (iss == 0) {
          mbar_arrive_expect_tx(&full[s], SKV);
          tma_load_2d(slot, &mkv, &full[s], 768 + qcol, samp * L + n * BN);                        // k: BN rows
        } else if (iss == 1) {
          mbar_arrive_expect_tx(&full[s], SKV);
          tma_load_2d(slot + SKV, &mkv, &full[s], 1536 + qcol, samp * L + n * BN);                 // v: BN rows
        } else {
          mbar_arrive_expect_tx(&full[s], SB);                      // the bias bytes arrive from rank 0's multicast
          if (rank == (uint32_t)(n % CLS))                          // rotate the issuer: one CTA issuing every block's
            tma_load_mc(slot + 2 * SKV, &mbias, &full[s], n * BN, brow0 + head * L + m0, (1u << CLS) - 1);
        }
      }
    }
    __syncwarp();
    cluster_arrive_relaxed(); cluster_wait();
    return;
  }

  // ---- consumer warpgroups: 64 query rows each
  const int lane = tid & 31, warp = (tid >> 5) & 3;
  const int r0 = warp * 16 + (lane >> 2), cb = 2 * (lane & 3);
  float acc[24];
#pragma unroll
  for (int i = 0; i < 24; ++i) acc[i] = 0.f;
#if !MAXLESS
  float m_i[2] = {-INFINITY, -INFINITY};
#endif
  float l_i[2] = {0.f, 0.f};
  if (LSUM) {                                                       // every byte 1.0 in bf16, so the swizzle cannot matter
#pragma unroll
    for (int i = 0; i < SONES / 4 / (128 * NWG); ++i)
      *reinterpret_cast<uint32_t*>(sones + 4 * (tid + i * 128 * NWG)) = 0x3f803f80u;
    named_bar_sync(5, 128 * NWG);
  }
  mbar_wait(qbar, 0);
  const uint32_t sq = smem_u32(sm) + (QREG ? OFF_ST + QSTAGE : 0) + wg * 8192;
  const uint64_t dOnes = LSUM ? dmn(smem_u32(sones), 0) : 0;
  float lacc[4] = {0.f, 0.f, 0.f, 0.f};
  // One offset per row, from the first key block, subtracted in every block: the same number the running max would
  // give (softmax is shift-invariant) with none of its per-block rescaling, and it keeps ex2 in range. Floored so a
  // row whose first key block is all masked cannot leave an offset that overflows later blocks.
  float m_off[2] = {0.f, 0.f};
  uint32_t qr[DH / 16][4];                                          // the A operand, straight out of shared memory
  if (QREG) {
#pragma unroll
    for (int ks = 0; ks < DH / 16; ++ks)
      ldsm_x4(qr[ks], sq + sw128(warp * 16 + 8 * ((lane >> 3) & 1) + (lane & 7), (16 * ks + 8 * (lane >> 4)) * 2));
    // The arrive must not overtake the ldmatrix reads it releases: make it data-dependent on their destinations
    // (zero_dep is 0 at run time but opaque to ptxas), or the producer refills slot 0 while they are still in flight.
    const uint32_t dep = qr[0][0] ^ qr[DH / 16 - 1][3];
    if ((tid & 31) == 0) mbar_arrive_dep(qdone, zero_dep(dep));      // slot 0 is free again before the producer fills it
  }

  // One block's softmax runs under the previous block's PV: the two matmuls of iteration n are committed as
  // QK(n+1) then PV(n), so wgmma_wait<1> at the top retires QK(n+1) while PV(n) is still in flight. The key loop is
  // unrolled by two and each half names its own score buffer: with a dynamic index ptxas cannot tell the buffer being
  // read from the one an in-flight wgmma writes, and injects a warpgroup.wait that serializes the pipeline (C7514).
  float sc0[32];
#if PIPE
  float sc1[32];
#endif
  uint32_t pa[BN / 16][4];
  const uint32_t sbase = smem_u32(sm) + OFF_ST;
  // One descriptor per slot per operand, built once: smem_desc is half a dozen shifts and ors, and a k-step only moves
  // the encoded address (32 bytes -> +2 for K-major, 2048 -> +128 for the MN-major V).
  uint64_t dK[DHOIST ? STAGES : 1], dV[DHOIST ? STAGES : 1];
#pragma unroll
  for (int i = 0; i < (DHOIST ? STAGES : 1); ++i) {
    dK[i] = dsw(sbase + i * ST_BYTES);
    dV[i] = dmn(sbase + i * ST_BYTES + SKV, 0);
  }

  auto qk_into = [&](float* dst, int n) {                           // wait for block n's tiles, issue its QK
    const int sn = n % STAGES;
    mbar_wait(&full[sn], (n / STAGES) & 1);
    if (FLOOR) return;
    if (BACC) {                                                     // seed with the bias; the wgmma accumulates onto it
      const uint8_t* sbn = reinterpret_cast<const uint8_t*>(sm) + OFF_ST + sn * ST_BYTES + 2 * SKV + wg * 8192;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint32_t bm[2][4];
#pragma unroll
        for (int half = 0; half < 2; ++half)
          ldsm_x4(bm[half], smem_u32(sbn) + sw128(16 * warp + 8 * h + (lane & 7), 8 * (4 * half + (lane >> 3)) * 2));
#pragma unroll
        for (int j = 0; j < 8; ++j) {                                // seeded with bias - offset, so ex2 needs no shift
          const float2 b = bf2f(bm[j >> 2][j & 3]);
          dst[4 * j + 2 * h] = MOFF ? b.x - m_off[h] : b.x;
          dst[4 * j + 2 * h + 1] = MOFF ? b.y - m_off[h] : b.y;
        }
      }
    }
    wgmma_fence();
#pragma unroll
    for (int ks = 0; ks < DH / 16; ++ks) {
      const uint64_t bk = DHOIST ? dK[sn] + ks * 2 : dsw(sbase + sn * ST_BYTES + ks * 32);
      if (QREG) mma_s_rs(dst, qr[ks], bk, BACC || ks != 0);
      else mma_s(dst, dsw(sq + ks * 32), bk, BACC || ks != 0);
    }
    wgmma_commit();
  };
  auto softmax_pv = [&](float* sc, int n, int leave) {
    const int sl = n % STAGES;
    const uint32_t slot = sbase + sl * ST_BYTES;
    if (FLOOR) {                                                    // floor mode: touch one word of each tile, no math
      acc[0] += __int_as_float(*reinterpret_cast<const int*>(reinterpret_cast<const uint8_t*>(sm) + OFF_ST
                                                            + sl * ST_BYTES + (tid & 31) * 4));
      return;
    }
    if (leave == 0) wgmma_wait<0>(); else wgmma_wait<1>();          // retire this block's QK, leave later groups running
#pragma unroll
    for (int i = 0; i < 32; ++i) fence_reg(sc[i]);
    const uint8_t* sb = reinterpret_cast<const uint8_t*>(sm) + OFF_ST + sl * ST_BYTES + 2 * SKV + wg * 8192;
#if !MAXLESS
    float alpha[2];
#endif
    float lsum[2] = {0.f, 0.f};
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int rr = r0 + 8 * h;
      float ssum = 0.f;
#if MAXLESS
      // No running max: softmax is shift-invariant, so this is the same number, and it drops the max reduction, the
      // rescale factor and the 24 FMAs that rescale the accumulator -- the bulk of the per-element softmax work here.
#if BACC
      if (MOFF && n == 0) {                                       // the first block pays for the offset it sets
        float mx = -INFINITY;
#pragma unroll
        for (int j = 0; j < 8; ++j) mx = fmaxf(mx, fmaxf(sc[4 * j + 2 * h], sc[4 * j + 2 * h + 1]));
        m_off[h] = fmaxf(qmax(mx), -60.f);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          sc[4 * j + 2 * h] -= m_off[h];
          sc[4 * j + 2 * h + 1] -= m_off[h];
        }
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {                                 // bias and offset are both in the accumulator
        const float p0 = ex2(sc[4 * j + 2 * h]), p1 = ex2(sc[4 * j + 2 * h + 1]);
        if (!LSUM) ssum += p0 + p1;
        const __nv_bfloat162 pk = __floats2bfloat162_rn(p0, p1);
        pa[j >> 1][2 * (j & 1) + h] = *reinterpret_cast<const uint32_t*>(&pk);
      }
#elif BLDM && !SOFTPV
      // ldmatrix hands each thread exactly the (row, column pair) the accumulator holds, four column groups at a time.
      uint32_t bm[2][4];
#pragma unroll
      for (int half = 0; half < 2; ++half)
        ldsm_x4(bm[half], smem_u32(sb) + sw128(16 * warp + 8 * h + (lane & 7), 8 * (4 * half + (lane >> 3)) * 2));
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float2 b = bf2f(bm[j >> 2][j & 3]);
        const float p0 = ex2(sc[4 * j + 2 * h] + b.x), p1 = ex2(sc[4 * j + 2 * h + 1] + b.y);
        ssum += p0 + p1;
        const __nv_bfloat162 pk = __floats2bfloat162_rn(p0, p1);
        pa[j >> 1][2 * (j & 1) + h] = *reinterpret_cast<const uint32_t*>(&pk);
      }
#else
#if SOFTPV
      if (h == 0) {                                                 // both halves of a k-step, then its wgmma
#pragma unroll
        for (int ks = 0; ks < BN / 16; ++ks) {
          uint32_t a4[4];
#pragma unroll
          for (int t = 0; t < 4; ++t) {                             // t = (column pair, row half) of this k-step
            const int j = 2 * ks + (t >> 1), hh = t & 1;
            const float2 b = bf2f(*reinterpret_cast<const uint32_t*>(sb + sw128(r0 + 8 * hh, (j * 8 + cb) * 2)));
            const float p0 = ex2(sc[4 * j + 2 * hh] + b.x), p1 = ex2(sc[4 * j + 2 * hh + 1] + b.y);
            lsum[hh] += p0 + p1;
            const __nv_bfloat162 pk = __floats2bfloat162_rn(p0, p1);
            a4[2 * (t >> 1) + hh] = *reinterpret_cast<const uint32_t*>(&pk);
          }
          wgmma_fence();
#pragma unroll
          for (int i = 0; i < 24; ++i) fence_reg(acc[i]);
          mma_o(acc, a4, dmn(slot + SKV, ks));                      // starts while the next k-step is still in ex2
        }
        wgmma_commit();
      }
      ssum = 0.f;
#else
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float2 b = bf2f(*reinterpret_cast<const uint32_t*>(sb + sw128(rr, (j * 8 + cb) * 2)));
        const float p0 = ex2(sc[4 * j + 2 * h] + b.x), p1 = ex2(sc[4 * j + 2 * h + 1] + b.y);
        ssum += p0 + p1;
        const __nv_bfloat162 pk = __floats2bfloat162_rn(p0, p1);
        pa[j >> 1][2 * (j & 1) + h] = *reinterpret_cast<const uint32_t*>(&pk);
      }
#endif
#endif
      l_i[h] += ssum + lsum[h];                                   // the quad reduction is deferred to the epilogue
#else
      float mx = -INFINITY;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float2 b = bf2f(*reinterpret_cast<const uint32_t*>(sb + sw128(rr, (j * 8 + cb) * 2)));
        sc[4 * j + 2 * h] += b.x;
        sc[4 * j + 2 * h + 1] += b.y;
        mx = fmaxf(mx, fmaxf(sc[4 * j + 2 * h], sc[4 * j + 2 * h + 1]));
      }
      mx = fmaxf(qmax(mx), -1e38f);
      const float m_new = fmaxf(m_i[h], mx);
      alpha[h] = ex2(m_i[h] - m_new);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float p0 = ex2(sc[4 * j + 2 * h] - m_new), p1 = ex2(sc[4 * j + 2 * h + 1] - m_new);
        ssum += p0 + p1;
        const __nv_bfloat162 pk = __floats2bfloat162_rn(p0, p1);    // both halves: .x of the raw struct is the low one
        pa[j >> 1][2 * (j & 1) + h] = *reinterpret_cast<const uint32_t*>(&pk);
      }
      l_i[h] = l_i[h] * alpha[h] + qsum(ssum);
      m_i[h] = m_new;
#endif
    }
    if (PIPE == 1) wgmma_wait<0>();                                 // PV(n-1) has landed in acc
    if (PIPE == 1 && n > 0 && (tid & 31) == 0) {
      const int sp = (n - 1) % STAGES;
#pragma unroll
      for (int k = 0; k < CLS; ++k) mbar_arrive_remote(&empty[sp], k);
    }
#if !MAXLESS
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int j = 0; j < 6; ++j) {
        acc[4 * j + 2 * h] *= alpha[h];
        acc[4 * j + 2 * h + 1] *= alpha[h];
      }
#endif
    if (!(MAXLESS && SOFTPV)) {                                     // the interleaved path already issued its wgmmas
      wgmma_fence();
#pragma unroll
      for (int i = 0; i < 24; ++i) fence_reg(acc[i]);
#pragma unroll
      for (int ks = 0; ks < BN / 16; ++ks) {
        mma_o(acc, pa[ks], DHOIST ? dV[sl] + ks * 128 : dmn(slot + SKV, ks));
        if (LSUM) mma_l(lacc, pa[ks], dOnes + ks * 128);
      }
      wgmma_commit();
    }
  };

#if PIPE == 1
  qk_into(sc0, 0);
  for (int n = 0; n < nblocks; n += 2) {
    if (n + 1 < nblocks) qk_into(sc1, n + 1);
    softmax_pv(sc0, n, n == 0 ? 0 : 1);
    if (n + 1 < nblocks) {
      if (n + 2 < nblocks) qk_into(sc0, n + 2);
      softmax_pv(sc1, n + 1, 1);
    }
  }
#elif PIPE == 2
  // Without a running max the key blocks are independent -- nothing is carried but the sums -- so two can be in flight:
  // softmax(n) runs under QK(n+1), and PV(n) under softmax(n+1).
  for (int n = 0; n < nblocks; n += 2) {
    const bool two = n + 1 < nblocks;
    qk_into(sc0, n);
    if (two) qk_into(sc1, n + 1);
    softmax_pv(sc0, n, two ? 1 : 0);
    if (two) softmax_pv(sc1, n + 1, 1);
    wgmma_wait<0>();
    if ((tid & 31) == 0)
#pragma unroll
      for (int k = 0; k < CLS; ++k) {
        mbar_arrive_remote(&empty[n % STAGES], k);
        if (two) mbar_arrive_remote(&empty[(n + 1) % STAGES], k);
      }
  }
#else
  // Ping-pong: warpgroup 1 waits until warpgroup 0 has its first key block through the MMA pipe, so from then on one
  // warpgroup's softmax (MUFU, ALU) runs under the other's wgmma instead of both idling the tensor cores together.
  if (PONG && NWG == 2 && wg == 1) named_bar_sync(9, 256);
  for (int n = 0; n < nblocks; ++n) {                               // QK, softmax, PV, release
    qk_into(sc0, n);
    if (PONG && NWG == 2 && wg == 0 && n == 0) named_bar_arrive(9, 256);
    softmax_pv(sc0, n, 0);
    if (!FLOOR) wgmma_wait<0>();
    if ((tid & 31) == 0)
#pragma unroll
      for (int k = 0; k < CLS; ++k) mbar_arrive_remote(&empty[n % STAGES], k);
  }
#endif
  if (!FLOOR) wgmma_wait<0>();
#pragma unroll
  for (int i = 0; i < 24; ++i) fence_reg(acc[i]);
#if PIPE == 1
  if ((tid & 31) == 0) {
    const int sp = (nblocks - 1) % STAGES;
#pragma unroll
    for (int k = 0; k < CLS; ++k) mbar_arrive_remote(&empty[sp], k);
  }
#endif

  // ---- epilogue: out = sigmoid(g) * o / l, over q
  const float ld0 = LSUM ? lacc[0] : (MAXLESS ? qsum(l_i[0]) : l_i[0]);
  const float ld1 = LSUM ? lacc[2] : (MAXLESS ? qsum(l_i[1]) : l_i[1]);
  const float inv[2] = {1.f / fmaxf(ld0, 1e-30f), 1.f / fmaxf(ld1, 1e-30f)};
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const size_t grow = (size_t)(row0 + wg * 64 + r0 + 8 * h) * (4 * 768) + 2304 + qcol;
    const size_t orow = (size_t)(row0 + wg * 64 + r0 + 8 * h) * (4 * 768) + qcol;
#pragma unroll
    for (int j = 0; j < 6; ++j) {                                   // 48 columns, two per thread per group
      const int c = j * 8 + cb;
      const float2 g = bf2f(*reinterpret_cast<const uint32_t*>(OUT + grow + c));
      const float o0 = acc[4 * j + 2 * h] * inv[h] * sigmoid_t(g.x);
      const float o1 = acc[4 * j + 2 * h + 1] * inv[h] * sigmoid_t(g.y);
      *reinterpret_cast<__nv_bfloat162*>(OUT + orow + c) = __floats2bfloat162_rn(o0, o1);
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
const CUtensorMap& tile_map(void* ptr, uint64_t rows, uint64_t cols, uint64_t stride, uint32_t bi, uint32_t bo) {
  static std::map<std::array<uint64_t, 6>, CUtensorMap> cache;
  static std::mutex lock;
  const std::array<uint64_t, 6> key{reinterpret_cast<uint64_t>(ptr), rows, cols, stride, bi, bo};
  std::lock_guard<std::mutex> g(lock);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  CUtensorMap map{};
  const cuuint64_t dims[2] = {cols, rows};
  const cuuint64_t strides[1] = {stride * 2};
  const cuuint32_t box[2] = {bi, bo}, elem[2] = {1, 1};
  TORCH_CHECK(encoder()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, dims, strides, box, elem,
                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS, "encode failed");
  return cache.emplace(key, map).first->second;
}

template <int CLS>
void launch(torch::Tensor& qkvg, torch::Tensor& bias, int64_t block, int L, int H, int S, float* dbg) {
  static_assert(CLS >= 1, "cluster size");
  const size_t smem = 1024 + SQ + STAGES * ST_BYTES + SONES + 256;
  auto kern = attn_kernel<CLS>;
  static bool once = [&] { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); return true; }();
  (void)once;
  const int mt = L / QM;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(mt * H * (CLS > 1 ? CLS : S));
  cfg.blockDim = dim3(128 * NWG + (PWARP ? 32 : 128));
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = CLS; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, kern, tile_map(qkvg.data_ptr(), S * L, 4 * 768, 4 * 768, QW, QM),
                                 tile_map(qkvg.data_ptr(), S * L, 4 * 768, 4 * 768, QW, BN),
                                 tile_map(bias.data_ptr(), bias.size(0) * L, L, L, BN, QM),
                                 reinterpret_cast<__nv_bfloat16*>(qkvg.data_ptr()), L, H, (int)block * H * L, mt, S, dbg)
              == cudaSuccess, "launch failed");
}
}  // namespace

// qkvg [S*L, 4*768] bf16 (q|k|v|g), bias [nb*H, L, L] bf16. Writes the gated attention output over q.
void attn_core(torch::Tensor qkvg, torch::Tensor bias, int64_t block, int64_t S, int64_t H,
               c10::optional<torch::Tensor> dbg) {
  const int L = bias.size(1);
  TORCH_CHECK(qkvg.is_contiguous() && qkvg.scalar_type() == torch::kBFloat16 && qkvg.size(1) == 4 * 768, "qkvg layout");
  TORCH_CHECK(bias.is_contiguous() && bias.scalar_type() == torch::kBFloat16 && bias.size(2) == L, "bias layout");
  TORCH_CHECK(L % QM == 0, "L must be a multiple of the query tile");
  // No cluster by default: the bias multicast moves 95 MB of L2 traffic a block and not one microsecond, because the
  // kernel is issue-bound, and a cluster constrains where the CTAs can be placed. ATTN_MC=1 turns it back on.
  if (!getenv("ATTN_MC")) { launch<1>(qkvg, bias, block, L, (int)H, (int)S, dbg ? dbg->data_ptr<float>() : nullptr); return; }
  switch (S) {
    case 4: launch<4>(qkvg, bias, block, L, (int)H, 4, dbg ? dbg->data_ptr<float>() : nullptr); break;
    case 5: launch<5>(qkvg, bias, block, L, (int)H, 5, dbg ? dbg->data_ptr<float>() : nullptr); break;
    case 2: launch<2>(qkvg, bias, block, L, (int)H, 2, dbg ? dbg->data_ptr<float>() : nullptr); break;
    case 1: launch<1>(qkvg, bias, block, L, (int)H, 1, dbg ? dbg->data_ptr<float>() : nullptr); break;
    default: TORCH_CHECK(false, "S must be 1, 2, 4 or 5");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("attn_core", &attn_core); }
