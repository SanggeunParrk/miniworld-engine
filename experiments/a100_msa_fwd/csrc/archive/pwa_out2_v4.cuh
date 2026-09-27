// pwa_out2.cuh -- PWA split path with the gate moved here, K_O2:
//   out[t, :] = msa[t, :] + sum_h (sigmoid(y[t, :] Wg_h'^T + bg_h) .* o[t, 32 h ..]) Wo_h^T,   o from pwa_ctr (-DPWA_C_NOGATE).
// The gate GEMM (64 -> 256 per token) was the compute-bound contraction kernel's epilogue; here it fills the idle tensor time of a
// memory-bound kernel (o 512 B + y 128 B + msa 128 B in, 128 B out per token). Persistent CTA of PWA_O2_WARPS warps, Wg' (256 x 128 B),
// Wo (64 x 512 B) and bg resident in smem; a warp walks its own 16-token tiles.
// v4: o arrives in fragment order from pwa_ctr: one 16 B load per (row, head) per lane instead of four 4 B ones.
// v3: o never touches smem. Each lane loads its accumulator-layout o values (rows g / g + 8, d = 8 dt + 2 q of every head) straight
// into registers, in two 4-head banks: heads 4-7 of this tile are requested at the tile start and consumed after heads 0-3 compute,
// heads 0-3 of the NEXT tile are requested while 4-7 compute. The residual is requested at the tile start too. Only y (2 KB per tile)
// goes through a private 2-slot cp.async ring. That frees the smem for 2 warps per SMSP (v1: 4 warps / SM, one per SMSP, issue-latency
// bound -- 2.9 cycles per issue -- and the late residual load was 11% of its stall samples).
// (Rejected v2, archive/pwa_out2_v2.cuh: 8 warps sharing 64-token tiles, head halves reduced through smem: 590 vs 558 us at L768.)
#pragma once
#include "sm80_common.cuh"

namespace a100 {

struct PwaOut2Params {
  const __nv_bfloat16* msa;   // [T, 64]
  const __nv_bfloat16* o;     // [T, 256]
  const __nv_bfloat16* y;     // [T, 64] LN(msa) without affine
  const __nv_bfloat16* wg;    // [256, 64] gamma folded
  const float* bg;            // [256]
  const __nv_bfloat16* wo;    // [64, 256]
  __nv_bfloat16* out;         // [T, 64]
  int T;
};

#ifndef PWA_O2_WARPS_
#define PWA_O2_WARPS_ 8
#endif
constexpr int PWA_O2_WARPS = PWA_O2_WARPS_;
constexpr int PWA_O2_WG = 0, PWA_O2_WO = 256 * 128, PWA_O2_BG = PWA_O2_WO + 64 * 512, PWA_O2_W = PWA_O2_BG + 1024;
constexpr int PWA_O2_SLOT = 16 * 128;                                      // y 2 KB
constexpr int PWA_O2_SMEM = PWA_O2_W + PWA_O2_WARPS * 2 * PWA_O2_SLOT;

DEVI uint32_t swz128o2(int row, int gr) { return row * 128 + ((gr ^ (row & 7)) << 4); }
DEVI uint32_t swz512o2(int row, int gr) { return row * 512 + ((gr ^ (row & 7)) << 4); }
DEVI uint32_t ldg32(const void* p) { return __ldg(reinterpret_cast<const unsigned int*>(p)); }

__global__ void __launch_bounds__(PWA_O2_WARPS * 32, 1) pwa_out2_kernel(PwaOut2Params p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sb = smem_u32(smem);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, q = lane & 3, l4 = lane >> 4;
  for (int idx = tid; idx < 256 * 8; idx += PWA_O2_WARPS * 32) {
    const int row = idx >> 3, gr = idx & 7;
    cp_async16(sb + PWA_O2_WG + swz128o2(row, gr), p.wg + row * 64 + gr * 8);
    const int ro = idx >> 5, go = idx & 31;
    cp_async16(sb + PWA_O2_WO + swz512o2(ro, go), p.wo + ro * 256 + go * 8);
  }
  float* sbg = reinterpret_cast<float*>(smem + PWA_O2_BG);
  for (int idx = tid; idx < 256; idx += PWA_O2_WARPS * 32) sbg[idx] = p.bg[idx];
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  const int ntiles = p.T / 16, gw = blockIdx.x * PWA_O2_WARPS + warp, nw = gridDim.x * PWA_O2_WARPS;
  const uint32_t wslot = sb + PWA_O2_W + warp * 2 * PWA_O2_SLOT;
  auto issue_y = [&](int tile, int slot) {
    const __nv_bfloat16* src_y = p.y + (size_t)tile * 16 * 64;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int idx = lane + 32 * k, row = idx >> 3, gr = idx & 7;
      cp_async16(wslot + slot * PWA_O2_SLOT + swz128o2(row, gr), src_y + row * 64 + gr * 8);
    }
  };
  // o bank: 4 heads x 4 dt x 2 rows (u32 = two bf16 at d = 8 dt + 2 q). pwa_ctr stores each head in fragment order (8 q + 2 dt + e),
  // so a lane's four dt values of one row are one 16 B vector.
  auto load_o = [&](int tile, int h0, uint32_t (&ob)[4][4][2]) {
    const __nv_bfloat16* r0 = p.o + ((size_t)tile * 16 + g) * 256 + 8 * q;
#pragma unroll
    for (int hh = 0; hh < 4; ++hh)
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const uint4 v = __ldg(reinterpret_cast<const uint4*>(r0 + rr * 8 * 256 + 32 * (h0 + hh)));
        ob[hh][0][rr] = v.x; ob[hh][1][rr] = v.y; ob[hh][2][rr] = v.z; ob[hh][3][rr] = v.w;
      }
  };
  const uint32_t yrow = swz128o2(lane & 15, l4);                                               // ^ (kc << 5)
  const uint32_t gb = PWA_O2_WG + swz128o2((lane & 7) + (l4 << 3), (lane >> 3) & 1);           // + (32 h + 16 np) 128, ^ (kc << 5)
  const uint32_t obw = PWA_O2_WO + swz512o2((lane & 7) + (l4 << 3), (lane >> 3) & 1);          // + 16 np 512, ^ ((4 h + 2 kc) << 4)

  // one head: gate GEMM from yf, u = sigmoid(g) o (o from the register bank), out-projection into oacc
  auto head = [&](int h, const uint32_t (&yf)[4][4], const uint32_t (&ov)[4][2], float (&oacc)[8][4]) {
    float gacc[4][4];
#pragma unroll
    for (int dt = 0; dt < 4; ++dt) {
      const float2 b = *reinterpret_cast<const float2*>(sbg + 32 * h + 8 * dt + 2 * q);
      gacc[dt][0] = gacc[dt][2] = b.x;
      gacc[dt][1] = gacc[dt][3] = b.y;
    }
#pragma unroll
    for (int kc = 0; kc < 4; ++kc)
#pragma unroll
      for (int np = 0; np < 2; ++np) {
        uint32_t bf[4];
        ldsm_x4(bf, sb + ((gb + (32 * h + 16 * np) * 128) ^ (kc << 5)));
        mma16816(gacc[2 * np], yf[kc], bf[0], bf[1]);
        mma16816(gacc[2 * np + 1], yf[kc], bf[2], bf[3]);
      }
    uint32_t ua[2][4];
#pragma unroll
    for (int dt = 0; dt < 4; ++dt) {
      ua[dt >> 1][(dt & 1) * 2] = pack_bf16(sigmoid(gacc[dt][0]) * bf16lo(ov[dt][0]), sigmoid(gacc[dt][1]) * bf16hi(ov[dt][0]));
      ua[dt >> 1][(dt & 1) * 2 + 1] = pack_bf16(sigmoid(gacc[dt][2]) * bf16lo(ov[dt][1]), sigmoid(gacc[dt][3]) * bf16hi(ov[dt][1]));
    }
#pragma unroll
    for (int kc = 0; kc < 2; ++kc)
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t bf[4];
        ldsm_x4(bf, sb + ((obw + 16 * np * 512) ^ ((4 * h + 2 * kc) << 4)));
        mma16816(oacc[2 * np], ua[kc], bf[0], bf[1]);
        mma16816(oacc[2 * np + 1], ua[kc], bf[2], bf[3]);
      }
  };

  int tile = gw, slot = 0;
  uint32_t oa[4][4][2], obk[4][4][2];
  if (tile < ntiles) { issue_y(tile, 0); load_o(tile, 0, oa); }
  cp_async_commit();
  for (; tile < ntiles; tile += nw, slot ^= 1) {
    const int nxt = tile + nw;
    load_o(tile, 4, obk);                                   // heads 4-7 of this tile, consumed after heads 0-3
    uint32_t xr[8][2];                                      // residual, consumed at the end
    const __nv_bfloat16* xrow = p.msa + ((size_t)tile * 16 + g) * 64 + 2 * q;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) { xr[nt][0] = ldg32(xrow + 8 * nt); xr[nt][1] = ldg32(xrow + 8 * 64 + 8 * nt); }
    if (nxt < ntiles) issue_y(nxt, slot ^ 1);
    cp_async_commit();
    cp_async_wait<1>();
    __syncwarp();
    uint32_t yf[4][4];
#pragma unroll
    for (int kc = 0; kc < 4; ++kc) ldsm_x4(yf[kc], wslot + slot * PWA_O2_SLOT + (yrow ^ (kc << 5)));
    float oacc[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) oacc[nt][e] = 0.f;
#pragma unroll
    for (int hh = 0; hh < 4; ++hh) head(hh, yf, oa[hh], oacc);
    if (nxt < ntiles) load_o(nxt, 0, oa);                   // next tile's heads 0-3, in flight during heads 4-7
#pragma unroll
    for (int hh = 0; hh < 4; ++hh) head(4 + hh, yf, obk[hh], oacc);
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const size_t base = ((size_t)tile * 16 + g + 8 * hh) * 64;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int c = 8 * nt + 2 * q;
        stg32(p.out + base + c, pack_bf16(bf16lo(xr[nt][hh]) + oacc[nt][2 * hh], bf16hi(xr[nt][hh]) + oacc[nt][2 * hh + 1]));
      }
    }
    __syncwarp();        // the y slot is refilled by the next iteration's issue_y()
  }
  cp_async_wait<0>();
}

}  // namespace a100
