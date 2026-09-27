// opm_epilogue.cuh -- OPM K_E: out[i, j, :] = (W_out . vec(O_ij)) / n_ij + bias (+ residual), bf16.
// O is GEMM1's output [(i, d), (j, e)] (row stride 32 L); vec(O_ij)[d * 32 + e] = O[i 32 + d, j 32 + e], so the [i,j,d,e] permute is
// folded into the operand loads. n_ij = max(1, popc(bits_i & bits_j)). Dividing after the projection equals dividing before it
// (normalize_before_proj: the bias is added after the division).
// CTA: 128 pairs (4 i x 32 j) x 128 outputs, K = 1024 in 32 / DPS stages of DPS d each (A: 128 pair rows x DPS x 32 e, B: 128 W rows x
// DPS x 32 e; rows of DPS x 64 B, granules XOR-swizzled), NS-stage cp.async ring. 8 warps = 4 (32 pairs) x 2 (64 outputs).
// v2: DPS templated (v1 = 1: two k16 steps per barrier, barrier-stall bound), every copy / ldmatrix offset precomputed per thread.
// v4: i_base: O may be one i-chunk of GEMM1 (the chunked, stream-pipelined OPM keeps each chunk L2-resident).
// v3: accumulators staged through smem, residual / out as 16 B coalesced rows (OPM_E_DIRECT keeps the v2 4 B register stores).
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
  int i_base;                 // O holds the rows of i = i_base .. (a chunk of GEMM1; 0 for the whole O)
  int transposed;             // O is GEMM1 transposed, O^T = b^T a: [(j, e), (i, d)] -- the tile's row groups are j, its column groups i
                              // (W must then be permuted to [c][e CH + d]); n_ij is symmetric, so only the output pair swaps
};

template <int NS_, int DPS_>
struct OpmECfg {
  static constexpr int NS = NS_, DPS = DPS_, NTHR = 256, TI = 4, TJ = 32;
  static constexpr int ROWB = DPS * 64, GPR = DPS * 4;          // bytes / 16 B granules per smem row
  static constexpr int TILE = 128 * ROWB, STAGE = 2 * TILE;
  static constexpr int SMEM = NS * STAGE + 128 * 4;
  static constexpr int MINB = 2 * SMEM <= 166 * 1024 ? 2 : 1;
  static constexpr int NCP = 128 * GPR / NTHR;                   // copies per thread per operand per stage
  DEVI static uint32_t swz(int row, int gr) {
    return row * ROWB + ((DPS == 1 ? (gr ^ ((row >> 1) & 3)) : (gr ^ (row & 7))) << 4);
  }
};

template <class G>
__global__ void __launch_bounds__(G::NTHR, G::MINB) opm_epilogue_kernel(OpmEpilogueParams p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sb = smem_u32(smem);
  float* inv_n = reinterpret_cast<float*>(smem + G::NS * G::STAGE);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, q = lane & 3, wm = warp & 3, wn = warp >> 2, l4 = lane >> 4;
  const int nj = p.L / G::TJ;
  const int i0 = (blockIdx.x / nj) * G::TI, j0 = (blockIdx.x % nj) * G::TJ;
  const size_t ldo = (size_t)32 * p.L;

  if (tid < 128) {
    const int i = p.i_base + i0 + (tid >> 5), j = j0 + (tid & 31);
    const uint32_t* bi = p.bits + (size_t)i * p.nw;
    const uint32_t* bj = p.bits + (size_t)j * p.nw;
    int c = 0;
    for (int w = 0; w < p.nw; ++w) c += __popc(__ldg(bi + w) & __ldg(bj + w));
    inv_n[tid] = 1.f / (float)max(c, 1);
  }

  // copy k of this thread: smem row / granule; A source = O row (i0 + row / 32) 32 + d0 + gr / 4, column (j0 + row % 32) 32 + (gr % 4) 8
  uint32_t s_off[G::NCP];
  const __nv_bfloat16* srcA[G::NCP];
  const __nv_bfloat16* srcB[G::NCP];
#pragma unroll
  for (int k = 0; k < G::NCP; ++k) {
    const int idx = tid + G::NTHR * k, row = idx / G::GPR, gr = idx % G::GPR;
    s_off[k] = G::swz(row, gr);
    srcA[k] = p.o + (size_t)((i0 + (row >> 5)) * 32 + (gr >> 2)) * ldo + (size_t)(j0 + (row & 31)) * 32 + (gr & 3) * 8;
    srcB[k] = p.w + row * 1024 + gr * 8;
  }
  auto load = [&](int st, int stage) {        // stage st covers d = st DPS .. + DPS - 1
    const uint32_t sa = sb + stage * G::STAGE;
#pragma unroll
    for (int k = 0; k < G::NCP; ++k) {
      cp_async16(sa + s_off[k], srcA[k] + (size_t)st * G::DPS * ldo);
      cp_async16(sa + G::TILE + s_off[k], srcB[k] + st * G::DPS * 32);
    }
  };
  uint32_t a_off[2], b_off[4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) a_off[mt] = G::swz(32 * wm + 16 * mt + (lane & 15), l4);
#pragma unroll
  for (int np = 0; np < 4; ++np) b_off[np] = G::TILE + G::swz(64 * wn + 16 * np + (lane & 7) + (l4 << 3), (lane >> 3) & 1);

  float acc[2][8][4];
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int b = 0; b < 8; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;

  constexpr int NST = 32 / G::DPS;
#pragma unroll
  for (int s = 0; s < G::NS - 1; ++s) { load(s, s); cp_async_commit(); }

  int stage = 0, lstage = G::NS - 1;
  for (int st = 0; st < NST; ++st) {
    cp_async_wait<G::NS - 2>();
    __syncthreads();
    if (st + G::NS - 1 < NST) load(st + G::NS - 1, lstage);
    cp_async_commit();
    if (++lstage == G::NS) lstage = 0;
    const uint32_t sa = sb + stage * G::STAGE;
    if (++stage == G::NS) stage = 0;
#pragma unroll
    for (int kk = 0; kk < 2 * G::DPS; ++kk) {
      uint32_t af[2][4], bf[4][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) ldsm_x4(af[mt], sa + (a_off[mt] ^ (kk << 5)));
#pragma unroll
      for (int np = 0; np < 4; ++np) ldsm_x4(bf[np], sa + (b_off[np] ^ (kk << 5)));
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) mma16816(acc[mt][nt], af[mt], bf[nt >> 1][(nt & 1) * 2], bf[nt >> 1][(nt & 1) * 2 + 1]);
    }
  }
  cp_async_wait<0>();

#ifdef OPM_E_DIRECT
  // epilogue: rows p = 32 wm + 16 mt + g (+8), cols c = 64 wn + 8 nt + 2 q
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int pr = 32 * wm + 16 * mt + g + 8 * h;
      const int rg = p.i_base + i0 + (pr >> 5), cgp = j0 + (pr & 31);
      const int i = p.transposed ? cgp : rg, j = p.transposed ? rg : cgp;
      const float s = inv_n[pr];
      const size_t base = ((size_t)i * p.L + j) * 128;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int c = 64 * wn + 8 * nt + 2 * q;
        const float2 b = __ldg(reinterpret_cast<const float2*>(p.bias + c));
        float x0 = fmaf(acc[mt][nt][2 * h], s, b.x), x1 = fmaf(acc[mt][nt][2 * h + 1], s, b.y);
        if (p.res) {
          const uint32_t r = __ldg(reinterpret_cast<const unsigned int*>(p.res + base + c));
          x0 += bf16lo(r); x1 += bf16hi(r);
        }
        stg32(p.out + base + c, pack_bf16(x0, x1));
      }
    }
#else
  // epilogue (the H100 kernel's accumulator staging): acc / n + bias -> fp32 tile [128 pairs][128] in the idle ring (64 KB, 16 B
  // granules XOR-swizzled by row), then each pair row leaves as 256 B contiguous with the residual read the same way. The 32 pairs of
  // one i are 32 consecutive j: 8 KB contiguous in out / res.
  static_assert(G::NS * G::STAGE >= 128 * 512, "the fp32 staging tile must fit in the ring");
  __syncthreads();
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int pr = 32 * wm + 16 * mt + g + 8 * h;
      const float s = inv_n[pr];
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int c = 64 * wn + 8 * nt + 2 * q;
        const float2 b = __ldg(reinterpret_cast<const float2*>(p.bias + c));
        const uint32_t a = sb + pr * 512 + ((((c >> 2) ^ (pr & 7))) << 4) + (c & 3) * 4;
        asm volatile("st.shared.v2.f32 [%0], {%1, %2};\n" ::"r"(a), "f"(fmaf(acc[mt][nt][2 * h], s, b.x)),
                     "f"(fmaf(acc[mt][nt][2 * h + 1], s, b.y)) : "memory");
      }
    }
  __syncthreads();
  // thread: 8 channels (2 granules) of a row per step; 16 threads per row, 16 rows per pass, 8 passes
  const int cg = tid & 15, r0 = tid >> 4;
#pragma unroll 4
  for (int pass = 0; pass < 8; ++pass) {
    const int pr = r0 + 16 * pass;
    const int rg = p.i_base + i0 + (pr >> 5), cgp = j0 + (pr & 31);
    const size_t base = (p.transposed ? (size_t)cgp * p.L + rg : (size_t)rg * p.L + cgp) * 128 + cg * 8;
    const uint4 x0 = lds128(sb + pr * 512 + (((2 * cg) ^ (pr & 7)) << 4));
    const uint4 x1 = lds128(sb + pr * 512 + (((2 * cg + 1) ^ (pr & 7)) << 4));
    float v[8] = {__uint_as_float(x0.x), __uint_as_float(x0.y), __uint_as_float(x0.z), __uint_as_float(x0.w),
                  __uint_as_float(x1.x), __uint_as_float(x1.y), __uint_as_float(x1.z), __uint_as_float(x1.w)};
    if (p.res) {
      const uint4 r = __ldg(reinterpret_cast<const uint4*>(p.res + base));
      v[0] += bf16lo(r.x); v[1] += bf16hi(r.x); v[2] += bf16lo(r.y); v[3] += bf16hi(r.y);
      v[4] += bf16lo(r.z); v[5] += bf16hi(r.z); v[6] += bf16lo(r.w); v[7] += bf16hi(r.w);
    }
    stg128(p.out + base, make_uint4(pack_bf16(v[0], v[1]), pack_bf16(v[2], v[3]), pack_bf16(v[4], v[5]), pack_bf16(v[6], v[7])));
  }
#endif
}

// ---------------------------------------------------------------------------------------------------- 256-pair tile
// v5: CTA = 8 i x 32 j = 256 pairs x 128 outputs (8 warps of 64 pairs x 64), K = 1024 in 32 one-d stages through a 4-deep ring
// (A 256 x 64 B + B 128 x 64 B = 24 KB per stage), 1 CTA / SM. Twice the pairs per CTA halves W's L2 re-reads (v4: W's 256 KB per
// 128 pairs, as much L2 traffic as O itself) and doubles the mma between barriers. The fp32 output staging is done per output half
// (64 KB each) so the residual add stays a single rounding.
DEVI uint32_t swz64(int row, int gr) { return row * 64 + ((gr ^ ((row >> 1) & 3)) << 4); }   // 64 B rows
constexpr int OPM_EB_NS = 4, OPM_EB_TA = 256 * 64, OPM_EB_STAGE = OPM_EB_TA + 128 * 64, OPM_EB_SMEM = OPM_EB_NS * OPM_EB_STAGE + 256 * 4;

__global__ void __launch_bounds__(256, 1) opm_epilogue_big_kernel(OpmEpilogueParams p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sb = smem_u32(smem);
  float* inv_n = reinterpret_cast<float*>(smem + OPM_EB_NS * OPM_EB_STAGE);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, q = lane & 3, wm = warp & 3, wn = warp >> 2, l4 = lane >> 4;
  const int nj = p.L / 32;
  const int i0 = (blockIdx.x / nj) * 8, j0 = (blockIdx.x % nj) * 32;
  const size_t ldo = (size_t)32 * p.L;
  {
    const int i = p.i_base + i0 + (tid >> 5), j = j0 + (tid & 31);
    const uint32_t* bi = p.bits + (size_t)i * p.nw;
    const uint32_t* bj = p.bits + (size_t)j * p.nw;
    int c = 0;
    for (int w = 0; w < p.nw; ++w) c += __popc(__ldg(bi + w) & __ldg(bj + w));
    inv_n[tid] = 1.f / (float)max(c, 1);
  }
  // copies: A granule (tid & 3) of pair rows (tid >> 2) + 64 k, k < 4; B granule (tid & 3) of W rows (tid >> 2) + 64 k, k < 2
  const int r0 = tid >> 2, gr = tid & 3;
  const uint32_t s_off = swz64(r0, gr);                        // + 64 k rows keeps (row >> 1) & 3
  const __nv_bfloat16* srcA = p.o + (size_t)((i0 + (r0 >> 5)) * 32) * ldo + (size_t)(j0 + (r0 & 31)) * 32 + gr * 8;
  const __nv_bfloat16* srcB = p.w + r0 * 1024 + gr * 8;
  auto load = [&](int d, int stage) {
    const uint32_t sa = sb + stage * OPM_EB_STAGE;
#pragma unroll
    for (int k = 0; k < 4; ++k) cp_async16(sa + s_off + k * 64 * 64, srcA + (size_t)(2 * k * 32 + d) * ldo);   // rows + 64 k: i + 2 k
#pragma unroll
    for (int k = 0; k < 2; ++k) cp_async16(sa + OPM_EB_TA + s_off + k * 64 * 64, srcB + k * 64 * 1024 + d * 32);
  };
  uint32_t a_off[4], b_off[4];
#pragma unroll
  for (int mt = 0; mt < 4; ++mt) a_off[mt] = swz64(64 * wm + 16 * mt + (lane & 15), l4);
#pragma unroll
  for (int np = 0; np < 4; ++np) b_off[np] = OPM_EB_TA + swz64(64 * wn + 16 * np + (lane & 7) + (l4 << 3), (lane >> 3) & 1);
  float acc[4][8][4];
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < 8; ++b)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;
#pragma unroll
  for (int s = 0; s < OPM_EB_NS - 1; ++s) { load(s, s); cp_async_commit(); }
  int stage = 0, lstage = OPM_EB_NS - 1;
  for (int d = 0; d < 32; ++d) {
    cp_async_wait<OPM_EB_NS - 2>();
    __syncthreads();
    if (d + OPM_EB_NS - 1 < 32) load(d + OPM_EB_NS - 1, lstage);
    cp_async_commit();
    if (++lstage == OPM_EB_NS) lstage = 0;
    const uint32_t sa = sb + stage * OPM_EB_STAGE;
    if (++stage == OPM_EB_NS) stage = 0;
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      uint32_t af[4][4], bf[4][4];
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) ldsm_x4(af[mt], sa + (a_off[mt] ^ (kk << 5)));
#pragma unroll
      for (int np = 0; np < 4; ++np) ldsm_x4(bf[np], sa + (b_off[np] ^ (kk << 5)));
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) mma16816(acc[mt][nt], af[mt], bf[nt >> 1][(nt & 1) * 2], bf[nt >> 1][(nt & 1) * 2 + 1]);
    }
  }
  cp_async_wait<0>();
  // output per half: warps wn == hf stage acc / n + bias as fp32 [256 pairs][64] (256 B rows), then 128 B per pair row + residual
  for (int hf = 0; hf < 2; ++hf) {
    __syncthreads();
    if (wn == hf) {
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int pr = 64 * wm + 16 * mt + g + 8 * h;
          const float sc = inv_n[pr];
#pragma unroll
          for (int nt = 0; nt < 8; ++nt) {
            const int cl = 8 * nt + 2 * q;
            const float2 b = __ldg(reinterpret_cast<const float2*>(p.bias + 64 * hf + cl));
            const uint32_t a = sb + pr * 256 + ((((cl >> 2) ^ (pr & 7))) << 4) + (cl & 3) * 4;
            asm volatile("st.shared.v2.f32 [%0], {%1, %2};\n" ::"r"(a), "f"(fmaf(acc[mt][nt][2 * h], sc, b.x)),
                         "f"(fmaf(acc[mt][nt][2 * h + 1], sc, b.y)) : "memory");
          }
        }
    }
    __syncthreads();
    const int cg = tid & 7, rr = tid >> 3;
#pragma unroll 4
    for (int pass = 0; pass < 8; ++pass) {
      const int pr = rr + 32 * pass;
      const int rg = p.i_base + i0 + (pr >> 5), cgp = j0 + (pr & 31);
      const size_t base = (p.transposed ? (size_t)cgp * p.L + rg : (size_t)rg * p.L + cgp) * 128 + 64 * hf + cg * 8;
      const uint4 x0 = lds128(sb + pr * 256 + (((2 * cg) ^ (pr & 7)) << 4));
      const uint4 x1 = lds128(sb + pr * 256 + (((2 * cg + 1) ^ (pr & 7)) << 4));
      float v[8] = {__uint_as_float(x0.x), __uint_as_float(x0.y), __uint_as_float(x0.z), __uint_as_float(x0.w),
                    __uint_as_float(x1.x), __uint_as_float(x1.y), __uint_as_float(x1.z), __uint_as_float(x1.w)};
      if (p.res) {
        const uint4 r = __ldg(reinterpret_cast<const uint4*>(p.res + base));
        v[0] += bf16lo(r.x); v[1] += bf16hi(r.x); v[2] += bf16lo(r.y); v[3] += bf16hi(r.y);
        v[4] += bf16lo(r.z); v[5] += bf16hi(r.z); v[6] += bf16lo(r.w); v[7] += bf16hi(r.w);
      }
      stg128(p.out + base, make_uint4(pack_bf16(v[0], v[1]), pack_bf16(v[2], v[3]), pack_bf16(v[4], v[5]), pack_bf16(v[6], v[7])));
    }
  }
}

}  // namespace a100
