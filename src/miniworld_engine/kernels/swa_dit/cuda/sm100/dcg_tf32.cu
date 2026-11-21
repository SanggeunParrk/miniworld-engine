// dcg_tf32.cu — the modulation's c gradient of the few-tile fp32 SWA atom block backward as two small tcgen05 GEMMs on sm_100a (r3-swa
// part 19), in place of the cuBLAS dc = dmod Wmod GEMM (K = 6C) and the silu' launch after it, both on the backward's dependent chain:
//   early (off the chain, beside the attention backward):  E  = dmod[:, 2C:6C] Wmod[2C:6C]          (the FFN / gate_a columns: ready
//                                                                                                   after the FFN adaLN backward)
//   late  (after the qkvg adaLN backward):                 dc = (E + dmod[:, 0:2C] Wmod[0:2C]) silu'(c)  [+ PREV] [+ XADD]
// The late launch is K = 2C (one third of the GEMM) with the silu' (and the stack's running sum, mod_bwd_tf32.cu swa_mod_dsilu_acc_sm100's
// equations) in its epilogue. TF32 operands (dmod fp32 as the MMA reads it, Wmod fp32), fp32 accumulation; another sum order than one K = 6C
// GEMM (E is summed first): deterministic (no atomics, fixed order).
// One CTA per (128-row tile, NB-column split): warp 0 TMA producer (the first NST stages one lane each -- a TMA issue costs its thread ~600
// clocks; the Wmod boxes before griddepcontrol.wait), warp 1 TMEM allocator + MMA issuer, warps 4-7 the epilogue (thread = tile row = TMEM
// lane). A stage: dmod [128 rows][32 fp32] SW128 (K-major A) | Wmod [NB / 32 atoms][32 rows][32 fp32] 128-B swizzle with 32-B atoms
// (MN-major B, the 3-D map of dcx_tf32.cu).
// SPDX-License-Identifier: Apache-2.0
#include "sm100.cuh"
using namespace s100;

namespace dcg {
__host__ __device__ constexpr uint32_t idesc(int M, int N, int a_mn = 0, int b_mn = 0) {
  return (1u << 4) | (2u << 7) | (2u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) | ((uint32_t)(N >> 3) << 17) |
         ((uint32_t)(M >> 4) << 24);
}
DEVI uint64_t desc_mn32b(uint32_t saddr, uint32_t lbo) {
  return (uint64_t)((saddr >> 4) & 0x3FFFu) | ((uint64_t)((lbo >> 4) & 0x3FFFu) << 16) | ((uint64_t)(512 >> 4) << 32) |
         ((uint64_t)1 << 46) | ((uint64_t)1 << 61);
}
DEVI void mma(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p; }"
               :: "r"(d), "l"(a), "l"(b), "r"(id), "r"(acc) : "memory");
}
DEVI float sigm(float x) { return 1.f / (1.f + __expf(-x)); }
DEVI float4 ld4b(const uint16_t* p) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u), __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
}
DEVI float4 ld4(const void* p, size_t o, bool b16) {
  return b16 ? ld4b(reinterpret_cast<const uint16_t*>(p) + o) : *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(p) + o);
}
DEVI uint32_t bf2(float lo, float hi) { uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
}  // namespace dcg

constexpr int C = 128, NSTMAX = 12;                                    // ring slots: as many as fit (NB 32: 11, 64: 9, 128: 6)
constexpr int SA = 128 * 128;                                          // A [128][32] fp32 16 KB | B [NB / 32 atoms][32][32] fp32 (NB * 128 B)
constexpr int DATA = 225280, O_BAR = DATA, SMEM_BYTES = O_BAR + 256;   // 11 slots of NB 32
static_assert(SMEM_BYTES <= 232448, "shared memory");

struct Bars {
  uint64_t full[NSTMAX], empty[NSTMAX], accfull;
  uint32_t tmem;
};

// flags: bit 0 early (OUT = the raw fp32 product), bit 1 c bf16, bit 2 OUT bf16, bit 3 XADD bf16
// grid = tiles x ns; NB = 128 / ns columns a CTA; k0 = the first K column of dmod (= the first Wmod row), nk = K / 32 stages
extern "C" __global__ void __launch_bounds__(256, 1)
swa_dcg_tf32_sm100(const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mb, int k0, int nk, int ns,
                   const float* __restrict__ ACCE, const void* __restrict__ CIN, const float* __restrict__ PREV,
                   const void* __restrict__ XADD, void* __restrict__ OUT, int flags) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const uint32_t su = smem_u32(sm);
  Bars& B = *reinterpret_cast<Bars*>(sm + O_BAR);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int tile = (int)blockIdx.x / ns, split = (int)blockIdx.x % ns;
  const int NB = C / ns, n0 = split * NB, row0 = tile * 128;
  const uint32_t bbytes = (uint32_t)(NB * 32 * 4);
  const int SLOT = SA + (int)bbytes, NST = min(NSTMAX, DATA / SLOT);   // (SLOT: a multiple of 4 KB, the swizzle atoms stay aligned)
  const bool early = flags & 1, cb16 = flags & 2, ob16 = flags & 4, xb16 = flags & 8;

  if (tid == 0) {
    for (int i = 0; i < NSTMAX; ++i) { mbar_init(&B.full[i], 1); mbar_init(&B.empty[i], 1); }
    mbar_init(&B.accfull, 1);
    fence_barrier_init();
  }
  if (warp == 1) { tmem_alloc(smem_u32(&B.tmem), 128); tmem_relinquish(); }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = B.tmem;
  pdl_launch();

  if (warp == 0) {
    // ------------------------------------------------------------------------------------------ TMA producer
    if (lane < NST && lane < nk) {                                      // the first stages: one lane each, Wmod before the wait
      const int s = lane;
      mbar_expect_tx(&B.full[s], (uint32_t)SA + bbytes);
      tma_load_3d(su + s * SLOT + SA, &mb, &B.full[s], 0, k0 + 32 * s, n0 / 32);
      pdl_wait();                                                       // dmod comes from the kernels before
      tma_load_2d(su + s * SLOT, &ma, &B.full[s], k0 + 32 * s, row0);
    }
    __syncwarp();
    if (lane == 0) {
      for (int s = NST; s < nk; ++s) {
        const int slot = s % NST;
        mbar_wait(&B.empty[slot], ((s / NST) - 1) & 1);
        mbar_expect_tx(&B.full[slot], (uint32_t)SA + bbytes);
        tma_load_3d(su + slot * SLOT + SA, &mb, &B.full[slot], 0, k0 + 32 * s, n0 / 32);
        tma_load_2d(su + slot * SLOT, &ma, &B.full[slot], k0 + 32 * s, row0);
      }
    }
    __syncwarp();
  } else if (warp == 1) {
    // ------------------------------------------------------------------------------------------ MMA issuer
    const uint32_t id = dcg::idesc(128, NB, 0, 1);
    for (int s = 0; s < nk; ++s) {
      const int slot = s % NST;
      mbar_wait(&B.full[slot], (s / NST) & 1);
      tc_fence_after();
      if (elect_one()) {
        const uint32_t a = su + slot * SLOT, b = a + SA;
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
          dcg::mma(tmem, desc_k128(a) + (uint64_t)(2 * ks), dcg::desc_mn32b(b + (uint32_t)(8 * ks) * 128u, 32 * 128), id, (s | ks) ? 1u : 0u);
        tc_commit(&B.empty[slot]);
        if (s == nk - 1) tc_commit(&B.accfull);
      }
      __syncwarp();
    }
  } else if (warp >= 4) {
    // ------------------------------------------------------------------------------------------ epilogue: thread = tile row
    const int r = 32 * (warp & 3) + lane, row = row0 + r;
    pdl_wait();                                                         // (PREV / XADD / c / E: written before this launch's wait)
    const uint32_t trow = tmem + ((uint32_t)(32 * (warp & 3)) << 16);
    const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 1
    for (int c0 = 0; c0 < NB; c0 += 32) {
      const size_t ob = (size_t)row * C + n0 + c0;
      // the chunk's other inputs first, all independent loads in flight at once (chunk 0: under the GEMM, before the accumulator wait)
      float4 ev[8], cv[8], pv[8], xv[8];
      if (!early) {
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const size_t o = ob + 4 * q;
          ev[q] = ACCE != nullptr ? *reinterpret_cast<const float4*>(ACCE + o) : z4;
          cv[q] = dcg::ld4(CIN, o, cb16);
          pv[q] = PREV != nullptr ? *reinterpret_cast<const float4*>(PREV + o) : z4;
          xv[q] = XADD != nullptr ? dcg::ld4(XADD, o, xb16) : z4;
        }
      }
      if (c0 == 0) { mbar_wait(&B.accfull, 0); tc_fence_after(); }
      uint32_t v[32];
      tmem_ld32(trow + (uint32_t)c0, v);
      tmem_wait_ld();
      if (early) {
        float* o = reinterpret_cast<float*>(OUT) + ob;
#pragma unroll
        for (int q = 0; q < 8; ++q)
          *reinterpret_cast<uint4*>(o + 4 * q) = make_uint4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]);
        continue;
      }
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const size_t o = ob + 4 * q;
        float4 g = make_float4(__uint_as_float(v[4 * q]), __uint_as_float(v[4 * q + 1]), __uint_as_float(v[4 * q + 2]), __uint_as_float(v[4 * q + 3]));
        if (ACCE != nullptr) {                                          // E (the early columns) first, then this launch's product
          const float4 e = ev[q];
          g = make_float4(e.x + g.x, e.y + g.y, e.z + g.z, e.w + g.w);
        }
        const float4 c = cv[q];
        const float sx = dcg::sigm(c.x), sy = dcg::sigm(c.y), sz = dcg::sigm(c.z), sw = dcg::sigm(c.w);
        float4 d = make_float4(g.x * sx * (1.f + c.x * (1.f - sx)), g.y * sy * (1.f + c.y * (1.f - sy)), g.z * sz * (1.f + c.z * (1.f - sz)),
                               g.w * sw * (1.f + c.w * (1.f - sw)));
        if (PREV != nullptr) {
          const float4 p = pv[q];
          d = make_float4(p.x + d.x, p.y + d.y, p.z + d.z, p.w + d.w);
        }
        if (XADD != nullptr) {
          const float4 x = xv[q];
          d = make_float4(x.x + d.x, x.y + d.y, x.z + d.z, x.w + d.w);
        }
        if (ob16)
          *reinterpret_cast<uint2*>(reinterpret_cast<uint16_t*>(OUT) + o) = make_uint2(dcg::bf2(d.x, d.y), dcg::bf2(d.z, d.w));
        else
          *reinterpret_cast<float4*>(reinterpret_cast<float*>(OUT) + o) = d;
      }
    }
    tc_fence_before();
  }
  __syncthreads();
  if (warp == 1) { tc_fence_after(); tmem_dealloc(tmem, 128); }
}
