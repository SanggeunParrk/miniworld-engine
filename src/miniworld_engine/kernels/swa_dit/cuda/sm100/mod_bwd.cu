// mod_bwd.cu — backward of the SWA atom block's adaLN modulation mod = silu(c) Wmod^T (mod_fwd.cu) on sm_100a, two kernels:
//   swa_mod_bwd_dc_sm100: dc = silu'(c) (g Wmod)          (g = dmod fp32 [R, 768], c bf16 [R, 128], Wmod bf16 [768, 128]; dc bf16)
//   swa_mod_bwd_dw_sm100: dWmod = rn(g^T silu(c))          (bf16 [768, 128])
// g is not bf16-valued: it is split into three bf16 terms (g = h + m + l, 24 significant bits; each difference is exact), and the
// three bf16 MMAs per K step form fp32-class products (c, Wmod and silu(c) are bf16 already) with fp32 accumulation.
// Both kernels split their reduction over the CTAs of a cluster (grid x): each CTA leaves its fp32 partial [128][128] in shared
// memory; after a cluster barrier every CTA sends CTA k the rows k owns (one bulk shared::cta -> shared::cluster copy per peer, into
// a receive slot per sender), and CTA k sums them in rank order (deterministic) and writes the result. Warps 0-7 split / MMA (thread 0) / reduce, warp 8 the TMA producer (three stages; the split terms go to one
// buffer the MMA of the previous step must have read).
// dc: cluster of 4 over the 768 channels (192 each, 3 K chunks of 64) for 128 rows; the split g as three K-major SW128 A tiles, Wmod
//     as the MN-major B. Result rows: silu backward dy s (1 + x (1 - s)), s = 1 / (1 + exp(-x)), fp32 (as torch) -> bf16.
// dW: cluster of G over the rows, CTA (row group, channel tile of 128): D = dW^T tile [128 d][128 ch] += silu(c)^T g over 64 rows at a
//     time (silu(c) as the MN-major A, the split g as three MN-major B); result: dWmod (bf16, transposed).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;
#ifndef TRACE
#define TRACE 0                            // 1: CTA (0, 0) thread 0 stamps globaltimer into TRb (timing experiments)
#endif
DEVI unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define EVT(k) do { if (TRACE && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) TRb[k] = gtime(); } while (0)

constexpr int CH = 768, DC = 128;
constexpr uint32_t I_MM = idesc_bf16(128, 128, 0, 1), I_NN = idesc_bf16(128, 128, 1, 1);

// bulk copy of `bytes` of our shared memory to offset `dst` of CTA `rank`, completing on its mbarrier at our offset `bar`
DEVI void bulk_to_peer(uint32_t dst, uint32_t src, uint32_t bytes, uint64_t* bar, uint32_t rank) {
  uint32_t d, b;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(dst), "r"(rank));
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(b) : "r"(smem_u32(bar)), "r"(rank));
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(d), "r"(src), "r"(bytes), "r"(b)
               : "memory");
}
// fp32 partial [128][128]: 16-B chunk q of row r (XOR within groups of 8 chunks)
DEVI uint32_t swp(uint32_t r, uint32_t q) { return r * 512u + (((q & ~7u) | ((q ^ r) & 7u)) << 4); }
// TMEM [128 lanes][128 fp32] -> the partial buffer (256 threads: lane 32 (warp % 4) + lane, columns 64 (warp / 4) .. + 63)
DEVI void tmem_to_partial(uint32_t tmem, uint32_t buf, int tid) {
  const uint32_t r = 32 * ((tid >> 5) & 3) + (tid & 31);
#pragma unroll 1
  for (int q = 2 * (tid >> 7); q < 2 * (tid >> 7) + 2; ++q) {
    uint32_t v[32];
    tmem_ld32(tmem + ((r & ~31u) << 16) + 32 * q, v);
    tmem_wait_ld();
#pragma unroll
    for (int c = 0; c < 8; ++c) sts128(buf + swp(r, 8 * q + c), make_uint4(v[4 * c], v[4 * c + 1], v[4 * c + 2], v[4 * c + 3]));
  }
}
// three-term bf16 split of x0, x1 (pairs packed as bf16x2): x = h + m + l
DEVI void split3(float x0, float x1, uint32_t& h, uint32_t& m, uint32_t& l) {
  h = pack_bf16(x0, x1);
  const float r0 = x0 - bf16lo(h), r1 = x1 - bf16hi(h);
  m = pack_bf16(r0, r1);
  l = pack_bf16(r0 - bf16lo(m), r1 - bf16hi(m));
}

// ======================================================================================================================== dc
constexpr int NST = 3;
constexpr int D_ST = 49152, D_W = 32768, D_A = NST * D_ST;                 // stage: g fp32 2 x [128][32] | Wmod 2 x [64 k][64 n]; 3 A tiles
constexpr int D_BAR = D_A + 3 * 16384, D_SMEM = D_BAR + 128, D_RCV = 65536;  // receive slots [4 senders][32 rows][512 B] after the partial

extern "C" __global__ void __launch_bounds__(288, 1)
swa_mod_bwd_dc_sm100(const __grid_constant__ CUtensorMap mg, const __grid_constant__ CUtensorMap mw, const uint16_t* __restrict__ Cc,
                     uint16_t* __restrict__ DCo, unsigned long long* __restrict__ TRb) {
  EVT(0);
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + D_BAR);
  uint64_t* sfree = full + NST;
  uint64_t* afree = sfree + NST;
  uint64_t* rbar = afree + 1;
  uint32_t* tptr = reinterpret_cast<uint32_t*>(rbar + 1);
  const int tid = threadIdx.x, warp = tid >> 5, kx = blockIdx.x, r0 = blockIdx.y * 128;   // kx: channels 192 kx .. + 191
  constexpr int NK = CH / 64 / 4;
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&full[s], 1); mbar_init(&sfree[s], 1); }
    mbar_init(afree, 1); mbar_init(rbar, 1);
    fence_barrier_init();
  }
  if (warp == 0) { tmem_alloc(smem_u32(tptr), 128); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tptr;
  EVT(1);
  if (warp == 8) {
    if ((tid & 31) == 0)
      for (int kc = 0; kc < NK; ++kc) {
        const int s = kc % NST;
        const uint32_t st = su + s * D_ST;
        if (kc >= NST) mbar_wait(&sfree[s], ((kc - NST) / NST) & 1);
        mbar_expect_tx(&full[s], D_ST);
        const int k0 = 192 * kx + 64 * kc;
        for (int i = 0; i < 2; ++i) tma_load_2d(st + i * 16384, &mg, &full[s], k0 + 32 * i, r0);
        for (int j = 0; j < 2; ++j) tma_load_2d(st + D_W + j * 8192, &mw, &full[s], 64 * j, k0);
      }
  } else {
    for (int kc = 0; kc < NK; ++kc) {
      const int s = kc % NST;
      const uint32_t st = su + s * D_ST;
      mbar_wait(&full[s], (kc / NST) & 1);
      EVT(2 + kc);
      if (kc > 0) mbar_wait(afree, (kc - 1) & 1);                           // the MMAs of kc - 1 have read the A tiles
      // row tid % 128, channels 32 (tid / 128) .. + 31 -> three bf16 A tiles ([128 rows][64 k], K-major SW128)
      const int rw = tid & 127, hh = tid >> 7;
#pragma unroll
      for (int q = 4 * hh; q < 4 * hh + 4; ++q) {
        const uint32_t gt = st + hh * 16384;
        const uint4 u0 = lds128(gt + sw128(rw, 2 * (q & 3))), u1 = lds128(gt + sw128(rw, 2 * (q & 3) + 1));
        const float x[8] = {__uint_as_float(u0.x), __uint_as_float(u0.y), __uint_as_float(u0.z), __uint_as_float(u0.w),
                            __uint_as_float(u1.x), __uint_as_float(u1.y), __uint_as_float(u1.z), __uint_as_float(u1.w)};
        uint32_t h[4], m[4], l[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) split3(x[2 * e], x[2 * e + 1], h[e], m[e], l[e]);
        const uint32_t a = su + D_A + sw128(rw, q);
        sts128(a, make_uint4(h[0], h[1], h[2], h[3]));
        sts128(a + 16384, make_uint4(m[0], m[1], m[2], m[3]));
        sts128(a + 32768, make_uint4(l[0], l[1], l[2], l[3]));
      }
      fence_proxy_async();
      tc_fence_before();
      named_bar_sync(1, 256);
      tc_fence_after();
      if (tid == 0) {
        const uint64_t dw = desc_mn128(st + D_W, 8192);
#pragma unroll
        for (int sp = 0; sp < 3; ++sp)
#pragma unroll
          for (int ks = 0; ks < 4; ++ks)
            umma_ss(tmem, desc_k128(su + D_A + sp * 16384) + (uint64_t)(ks * 2), dw + (uint64_t)(ks * 128), I_MM, (kc | sp | ks) ? 1u : 0u);
        tc_commit(&sfree[s]);
        tc_commit(afree);
      }
    }
    mbar_wait(afree, (NK - 1) & 1);
    EVT(6);
    tc_fence_after();
    tmem_to_partial(tmem, su, tid);                                        // over stages 0-1 (every load consumed)
    fence_proxy_async();
    if (tid == 0) mbar_expect_tx(rbar, 3 * 16384);                         // the three peers' rows 32 kx .. + 31
    EVT(7);
  }
  cluster_sync();                                                          // every partial written, every receive armed
  if (tid == 0)
    for (int p = 0; p < 4; ++p)
      if (p != kx) bulk_to_peer(su + D_RCV + kx * 16384, su + p * 16384, 16384, rbar, p);
  mbar_wait(rbar, 0);
  EVT(8);
  if (tid < 256) {
    // rows 32 kx + tid / 8, columns 16 (tid % 8) .. + 15: the four partials (ranks 0-3) summed, then the silu backward
    const int rr = 32 * kx + (tid >> 3), cg = tid & 7;
    const size_t row = (size_t)(r0 + rr) * DC + 16 * cg;
    uint4 u[4][4];                                                         // rank p's rows: our partial (p = kx) or receive slot p
#pragma unroll
    for (int h = 0; h < 4; ++h)
#pragma unroll
      for (int p = 0; p < 4; ++p)
        u[h][p] = lds128(p == kx ? su + swp(rr, 4 * cg + h) : su + D_RCV + p * 16384 + swp(rr, 4 * cg + h) - 32 * kx * 512);
    const uint4* cr = reinterpret_cast<const uint4*>(Cc + row);
    const uint4 cu[2] = {__ldg(cr), __ldg(cr + 1)};
    float f[16];
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      float4 acc = make_float4(__uint_as_float(u[h][0].x), __uint_as_float(u[h][0].y), __uint_as_float(u[h][0].z), __uint_as_float(u[h][0].w));
#pragma unroll
      for (int p = 1; p < 4; ++p) {
        acc.x += __uint_as_float(u[h][p].x); acc.y += __uint_as_float(u[h][p].y); acc.z += __uint_as_float(u[h][p].z); acc.w += __uint_as_float(u[h][p].w);
      }
      f[4 * h] = acc.x; f[4 * h + 1] = acc.y; f[4 * h + 2] = acc.z; f[4 * h + 3] = acc.w;
    }
    uint32_t o[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const uint32_t cw = (e < 4) ? (&cu[0].x)[e] : (&cu[1].x)[e - 4];
      float dx[2];
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        const float x = t ? bf16hi(cw) : bf16lo(cw), dy = f[2 * e + t], sg = 1.f / (1.f + expf(-x));
        dx[t] = dy * sg * (1.f + x * (1.f - sg));
      }
      o[e] = pack_bf16(dx[0], dx[1]);
    }
    uint4* orow = reinterpret_cast<uint4*>(DCo + row);
    orow[0] = make_uint4(o[0], o[1], o[2], o[3]); orow[1] = make_uint4(o[4], o[5], o[6], o[7]);
  }
  EVT(9);
  cluster_sync();                                                          // our copies to the peers are complete
  EVT(10);
  tc_fence_before();
  __syncthreads();
  if (warp == 0) { tc_fence_after(); tmem_dealloc(tmem, 128); }
}

// ======================================================================================================================== dW
constexpr int W_ST = 49152, W_C = 32768, W_B = NST * W_ST;                 // stage: g fp32 4 x [64][32] | c -> silu(c) 2 x [64][64]; 3 B
static_assert(2 * W_ST >= 65536 && 2 * D_ST >= 65536, "the fp32 partial over stages 0-1");
constexpr int W_BAR = W_B + 3 * 16384, W_SMEM = W_BAR + 128, W_RCV = 65536;  // receive slots [G senders][128 / G rows][512 B]

extern "C" __global__ void __launch_bounds__(288, 1)
swa_mod_bwd_dw_sm100(const __grid_constant__ CUtensorMap mg, const __grid_constant__ CUtensorMap mc,
                     uint16_t* __restrict__ DW, int NS) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + W_BAR);
  uint64_t* sfree = full + NST;
  uint64_t* bfree = sfree + NST;
  uint64_t* rbar = bfree + 1;
  uint32_t* tptr = reinterpret_cast<uint32_t*>(rbar + 1);
  const int tid = threadIdx.x, warp = tid >> 5, gx = blockIdx.x, G = gridDim.x, ct = blockIdx.y, rb = gx * NS * 64;
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(&full[s], 1); mbar_init(&sfree[s], 1); }
    mbar_init(bfree, 1); mbar_init(rbar, 1);
    fence_barrier_init();
  }
  if (warp == 0) { tmem_alloc(smem_u32(tptr), 128); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tptr;
  if (warp == 8) {
    if ((tid & 31) == 0)
      for (int j = 0; j < NS; ++j) {
        const int s = j % NST;
        const uint32_t st = su + s * W_ST;
        if (j >= NST) mbar_wait(&sfree[s], ((j - NST) / NST) & 1);
        mbar_expect_tx(&full[s], W_ST);
        for (int i = 0; i < 4; ++i) tma_load_2d(st + i * 8192, &mg, &full[s], 128 * ct + 32 * i, rb + 64 * j);
        for (int i = 0; i < 2; ++i) tma_load_2d(st + W_C + i * 8192, &mc, &full[s], 64 * i, rb + 64 * j);
      }
  } else {
    const int k = tid & 63, qq = tid >> 6;                                 // row k of the 64, quarter qq of the channels / of d
    for (int j = 0; j < NS; ++j) {
      const int s = j % NST;
      const uint32_t st = su + s * W_ST;
      mbar_wait(&full[s], (j / NST) & 1);
      if (j > 0) mbar_wait(bfree, (j - 1) & 1);
      // g row k, channels 32 qq .. + 31 -> three bf16 B blocks ([64 k][64 n] SW128, MN-major)
#pragma unroll
      for (int q = 4 * (qq & 1); q < 4 * (qq & 1) + 4; ++q) {
        const uint32_t gt = st + qq * 8192;
        const uint4 u0 = lds128(gt + sw128(k, 2 * (q & 3))), u1 = lds128(gt + sw128(k, 2 * (q & 3) + 1));
        const float x[8] = {__uint_as_float(u0.x), __uint_as_float(u0.y), __uint_as_float(u0.z), __uint_as_float(u0.w),
                            __uint_as_float(u1.x), __uint_as_float(u1.y), __uint_as_float(u1.z), __uint_as_float(u1.w)};
        uint32_t h[4], m[4], l[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) split3(x[2 * e], x[2 * e + 1], h[e], m[e], l[e]);
        const uint32_t a = su + W_B + (qq >> 1) * 8192 + sw128(k, q);
        sts128(a, make_uint4(h[0], h[1], h[2], h[3]));
        sts128(a + 16384, make_uint4(m[0], m[1], m[2], m[3]));
        sts128(a + 32768, make_uint4(l[0], l[1], l[2], l[3]));
      }
      // silu(c) in place (bf16, the approximate sigmoid of mod_fwd.cu), row k, d 32 qq .. + 31
#pragma unroll
      for (int q = 4 * (qq & 1); q < 4 * (qq & 1) + 4; ++q) {
        const uint32_t a = st + W_C + (qq >> 1) * 8192 + sw128(k, q);
        const uint4 u = lds128(a);
        uint32_t w4[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float x0 = bf16lo(w4[e]), x1 = bf16hi(w4[e]);
          w4[e] = pack_bf16(x0 * sigmoid_kit(x0), x1 * sigmoid_kit(x1));
        }
        sts128(a, make_uint4(w4[0], w4[1], w4[2], w4[3]));
      }
      fence_proxy_async();
      tc_fence_before();
      named_bar_sync(1, 256);
      tc_fence_after();
      if (tid == 0) {
        const uint64_t da = desc_mn128(st + W_C, 8192);
#pragma unroll
        for (int sp = 0; sp < 3; ++sp)
#pragma unroll
          for (int ks = 0; ks < 4; ++ks)
            umma_ss(tmem, da + (uint64_t)(ks * 128), desc_mn128(su + W_B + sp * 16384, 8192) + (uint64_t)(ks * 128), I_NN, (j | sp | ks) ? 1u : 0u);
        tc_commit(&sfree[s]);
        tc_commit(bfree);
      }
    }
    mbar_wait(bfree, (NS - 1) & 1);                                        // every MMA done: the stages are free for the staging
    tc_fence_after();
    tmem_to_partial(tmem, su, tid);                                        // partial dW^T tile [128 d][128 ch]
    fence_proxy_async();
    if (tid == 0) mbar_expect_tx(rbar, (G - 1) * (128 / G) * 512);         // the peers' rows (128 / G) gx ..
  }
  const int rp = 128 / G;
  cluster_sync();
  if (tid == 0)
    for (int p = 0; p < G; ++p)
      if (p != gx) bulk_to_peer(su + W_RCV + gx * rp * 512, su + p * rp * 512, rp * 512, rbar, p);
  mbar_wait(rbar, 0);
  if (tid < 256) {
    // d rows (128 / G) gx .. : 256 / rp threads per row d, nq chunks of 4 channels each; ranks summed in order -> dWmod[ch][d]
    const int tpr = 256 / rp, nq = 32 / tpr, d = rp * gx + tid / tpr, q0 = (tid % tpr) * nq;
    for (int i = 0; i < nq; ++i) {
      const uint32_t off = swp(d, q0 + i) - rp * gx * 512;                 // within a receive slot
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      for (int p = 0; p < G; ++p) {                                        // rank order
        const uint4 u = lds128(p == gx ? su + swp(d, q0 + i) : su + W_RCV + p * rp * 512 + off);
        if (p == 0) { acc[0] = __uint_as_float(u.x); acc[1] = __uint_as_float(u.y); acc[2] = __uint_as_float(u.z); acc[3] = __uint_as_float(u.w); }
        else { acc[0] += __uint_as_float(u.x); acc[1] += __uint_as_float(u.y); acc[2] += __uint_as_float(u.z); acc[3] += __uint_as_float(u.w); }
      }
#pragma unroll
      for (int e = 0; e < 4; ++e) DW[(size_t)(128 * ct + 4 * (q0 + i) + e) * DC + d] = (uint16_t)(pack_bf16(acc[e], 0.f) & 0xffffu);
    }
  }
  cluster_sync();
  tc_fence_before();
  __syncthreads();
  if (warp == 0) { tc_fence_after(); tmem_dealloc(tmem, 128); }
}
