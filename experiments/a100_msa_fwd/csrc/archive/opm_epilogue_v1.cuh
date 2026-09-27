// opm_epilogue.cuh -- OPM K_E: out[i, j, :] = (W_out . vec(O_ij)) / n_ij + bias (+ residual), bf16.
// O is GEMM1's output [(i, d), (j, e)] (row stride 32 L); vec(O_ij)[d * 32 + e] = O[i 32 + d, j 32 + e], so the [i,j,d,e] permute is
// folded into the operand loads. n_ij = max(1, popc(bits_i & bits_j)). Dividing after the projection equals dividing before it
// (normalize_before_proj: the bias is added after the division).
// CTA: 128 pairs (4 i x 32 j) x 128 outputs, K = 1024 in 32 chunks (one d each: A 128 pair rows x 32 e, B 128 W rows x 32 e, both
// 64 B rows), NS-stage cp.async ring. 8 warps = 4 (32 pairs) x 2 (64 outputs).
#pragma once
#include "sm80_common.cuh"

namespace a100 {

struct OpmEpilogueParams {
  const __nv_bfloat16* o;     // [32 L, 32 L]
  const __nv_bfloat16* w;     // [128, 1024] to_out.weight
  const float* bias;          // [128]
  const uint32_t* bits;       // [L, nw]
  const __nv_bfloat16* res;   // [L, L, 128] or nullptr
  __nv_bfloat16* out;         // [L, L, 128]
  int L, nw;
};

template <int NS_>
struct OpmECfg {
  static constexpr int NS = NS_, NTHR = 256, TI = 4, TJ = 32;
  static constexpr int STAGE = 2 * 128 * 64;             // A + B, bytes
  static constexpr int SMEM = NS * STAGE + 128 * 4;
};

DEVI uint32_t swz64(int row, int gr) { return row * 64 + ((gr ^ ((row >> 1) & 3)) << 4); }

template <class G>
__global__ void __launch_bounds__(G::NTHR, 2) opm_epilogue_kernel(OpmEpilogueParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t sb = smem_u32(smem);
  float* inv_n = reinterpret_cast<float*>(smem + G::NS * G::STAGE);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, q = lane & 3, wm = warp & 3, wn = warp >> 2;
  const int nj = p.L / G::TJ;
  const int i0 = (blockIdx.x / nj) * G::TI, j0 = (blockIdx.x % nj) * G::TJ;
  const size_t ldo = (size_t)32 * p.L;

  if (tid < 128) {
    const int i = i0 + (tid >> 5), j = j0 + (tid & 31);
    const uint32_t* bi = p.bits + (size_t)i * p.nw;
    const uint32_t* bj = p.bits + (size_t)j * p.nw;
    int c = 0;
    for (int w = 0; w < p.nw; ++w) c += __popc(__ldg(bi + w) & __ldg(bj + w));
    inv_n[tid] = 1.f / (float)max(c, 1);
  }

  // copy k (0 / 1): idx = tid + 256 k over 512 granules of A, the same over B
  auto load = [&](int d, int stage) {
    const uint32_t sa = sb + stage * G::STAGE, sw = sa + 128 * 64;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int idx = tid + 256 * k, row = idx >> 2, gr = idx & 3;
      const __nv_bfloat16* srcA = p.o + (size_t)((i0 + (row >> 5)) * 32 + d) * ldo + (size_t)(j0 + (row & 31)) * 32 + gr * 8;
      cp_async16(sa + swz64(row, gr), srcA);
      cp_async16(sw + swz64(row, gr), p.w + row * 1024 + d * 32 + gr * 8);
    }
  };

  float acc[2][8][4];
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int b = 0; b < 8; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;

#pragma unroll
  for (int s = 0; s < G::NS - 1; ++s) { load(s, s); cp_async_commit(); }

  for (int d = 0; d < 32; ++d) {
    cp_async_wait<G::NS - 2>();
    __syncthreads();
    if (d + G::NS - 1 < 32) load(d + G::NS - 1, (d + G::NS - 1) % G::NS);
    cp_async_commit();
    const uint32_t sa = sb + (d % G::NS) * G::STAGE, sw = sa + 128 * 64;
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      uint32_t af[2][4], bf[4][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) ldsm_x4(af[mt], sa + swz64(32 * wm + 16 * mt + (lane & 15), 2 * kk + (lane >> 4)));
#pragma unroll
      for (int np = 0; np < 4; ++np)
        ldsm_x4(bf[np], sw + swz64(64 * wn + 16 * np + (lane & 7) + ((lane >> 4) << 3), 2 * kk + ((lane >> 3) & 1)));
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) mma16816(acc[mt][nt], af[mt], bf[nt >> 1][(nt & 1) * 2], bf[nt >> 1][(nt & 1) * 2 + 1]);
    }
  }
  cp_async_wait<0>();

  // epilogue: rows p = 32 wm + 16 mt + g (+8), cols c = 64 wn + 8 nt + 2 q (+1)
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int pr = 32 * wm + 16 * mt + g + 8 * h;
      const int i = i0 + (pr >> 5), j = j0 + (pr & 31);
      const float s = inv_n[pr];
      const size_t base = ((size_t)i * p.L + j) * 128;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int c = 64 * wn + 8 * nt + 2 * q;
        float x0 = fmaf(acc[mt][nt][2 * h], s, __ldg(p.bias + c)), x1 = fmaf(acc[mt][nt][2 * h + 1], s, __ldg(p.bias + c + 1));
        if (p.res) {
          const uint32_t r = __ldg(reinterpret_cast<const unsigned int*>(p.res + base + c));
          x0 += bf16lo(r); x1 += bf16hi(r);
        }
        stg32(p.out + base + c, pack_bf16(x0, x1));
      }
    }
}

}  // namespace a100
