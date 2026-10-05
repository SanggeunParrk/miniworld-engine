// relpos_sm80.cuh -- the backward of the ProteinMPNN relative-position embedding on A100 (sm_80): one gradient row per EDGE reduced into a [buckets, 16] table (and the same rows into the bias).
//
//   grad_table[b, :] = sum_{edges e: bucket[e] = b} grad[e, :]          grad_bias[:] = sum_e grad[e, :]
//
// At B = 16, T = 8192, K = 48 that is 6.3 M rows into 66, a third of them into the two clamp buckets: an atomic-per-element scatter (or F.embedding's sort-and-segment backward) is 5-30 ms.
// Here the scatter is a MATMUL against the one-hot of the bucket index, on the tensor cores: per 16 edges, mma.m16n8k16 with A = onehot^T [80 bucket rows x 16 edges] (exact 0 / 1 in bf16),
// B = the 16 x 16 gradient tile (bf16 gradients are exact operands; an fp32 gradient is split into three bf16 pieces hi + mid + lo = the fp32 value exactly: three mma), fp32 accumulation.
// The bias gradient rides in the table's last row (row 79 of the padded 80: all-ones A), so it costs nothing extra.  Buckets 0 .. 78 are served (the shipped table is 66).
//
// Deterministic by construction: every warp owns a contiguous chunk of 16-row steps and accumulates it in registers (the order inside a warp is fixed), the warps of a CTA are summed in warp order,
// the CTAs' partial tables in CTA order by a second tiny kernel: no atomics, so the gradient is bit-for-bit reproducible (the same property of the Triton kernel's fixed-order partial sum).
// Each warp streams its rows through a 4-stage cp.async ring (the gradient tile and the 16 int64 bucket ids per step) -- the kernel is a pure HBM stream: 40 B / edge for bf16.
#pragma once
#include "mpnn_common.cuh"

namespace mp80 {

constexpr int RP_ROWS = 80;                     // padded bucket rows: buckets 0 .. 78 and the bias row 79
constexpr int RP_W = 16;                        // the shipped channel width
constexpr int RP_TAB = RP_ROWS * RP_W;          // floats per partial table

struct RelposParams {
  const void* grad;                             // [rows][16] bf16 or fp32
  const int64_t* bucket;                        // [rows]
  float* partial;                               // [ctas][RP_TAB]
  int64_t rows;
  int nbuckets;
};

template <bool F32>
struct RelposCfg {
  static constexpr int NW = 8, NTHR = NW * 32, NSTG = 4, MINB = 3;
  static constexpr int GBYTES = F32 ? 16 * 80 : 16 * 32;           // the gradient tile of one 16-row step: fp32 rows are padded to 80 B (conflict-free fragment reads)
  static constexpr int STAGE = GBYTES + 128;                        // + 16 int64 bucket ids
  static constexpr int RING = NW * NSTG * STAGE;
  static constexpr int TAB = NW * RP_TAB * 4;                       // the warps' tables of the CTA combine
  static constexpr int SMEM = RING > TAB ? RING : TAB;
};

// bf16 tile: 16 rows x 32 B, granule g (16 B) of row r at granule g ^ ((r >> 2) & 1) -- the eight rows of an ldmatrix matrix hit eight distinct bank groups
DEVI uint32_t rp_swz(uint32_t row, uint32_t g) { return row * 32u + ((g ^ ((row >> 2) & 1u)) << 4); }

template <bool F32>
__global__ void __launch_bounds__(RelposCfg<F32>::NTHR, RelposCfg<F32>::MINB) relpos_kernel(const RelposParams p) {
  using C = RelposCfg<F32>;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int64_t nsteps = (p.rows + 15) >> 4;
  const int64_t wtot = (int64_t)gridDim.x * C::NW, wid = (int64_t)blockIdx.x * C::NW + warp;
  const int64_t s0 = nsteps * wid / wtot, s1 = nsteps * (wid + 1) / wtot;
  const int64_t nmine = s1 - s0;
  const uint32_t ring = sb + warp * (C::NSTG * C::STAGE);

  float acc[5][2][4];
#pragma unroll
  for (int mt = 0; mt < 5; ++mt)
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) { acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f; }
  const uint32_t ones = (g8 == 7) ? 0x3F803F80u : 0u;           // row 79 of the table (m-tile 4, the upper half of the lane group g8 = 7): every edge belongs to it

  auto issue = [&](int64_t step, int stage) {
    const uint32_t st = ring + stage * C::STAGE;
    const int64_t row0 = step * 16;
    if constexpr (!F32) {                                         // 512 B = 32 chunks: lane -> (row, half)
      const int r = lane >> 1, h = lane & 1;
      const bool ok = row0 + r < p.rows;
      cp_async16(st + rp_swz(r, h), reinterpret_cast<const bf*>(p.grad) + (ok ? (row0 + r) * 16 + h * 8 : 0), ok ? 16 : 0);
    } else {                                                      // 1 KB = 64 chunks: lane and lane + 32 -> (row, quarter), rows padded to 80 B
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int c = lane + 32 * i, r = c >> 2, qd = c & 3;
        const bool ok = row0 + r < p.rows;
        cp_async16(st + r * 80 + qd * 16, reinterpret_cast<const float*>(p.grad) + (ok ? (row0 + r) * 16 + qd * 4 : 0), ok ? 16 : 0);
      }
    }
    if (lane < 8) {                                               // 16 int64 = 8 chunks: rows 2 lane, 2 lane + 1
      const int64_t r = row0 + 2 * lane;
      const int64_t left = p.rows - r;
      const uint32_t bytes = left >= 2 ? 16u : (left == 1 ? 8u : 0u);
      cp_async16(st + C::GBYTES + lane * 16, p.bucket + (bytes ? r : 0), bytes);
    }
  };

#pragma unroll
  for (int i = 0; i < C::NSTG - 1; ++i) {
    if (i < nmine) issue(s0 + i, i);
    cp_async_commit();
  }
  for (int64_t i = 0; i < nmine; ++i) {
    cp_async_wait<C::NSTG - 2>();
    __syncwarp();
    const uint32_t st = ring + (int)(i % C::NSTG) * C::STAGE;
    // ---- consume the stage: the B fragments of the gradient tile (2 n8 tiles of 8 channels) and this lane's four bucket ids (rows 2 q4, 2 q4 + 1, 2 q4 + 8, 2 q4 + 9)
    uint32_t bp[3][2][2];                                         // fp32: [piece hi / mid / lo][n tile][b0, b1]; bf16 uses bfr
    uint32_t bfr[4];
    float fv[2][4];                                               // fp32: rows 2 q4 and 2 q4 + 1 (+ 8) of channel g8 (+ 8 for the second n tile)
    if constexpr (!F32) {
      const int mi = lane >> 3, row = 8 * (mi & 1) + (lane & 7);
      ldsm_x4_t(bfr, st + rp_swz(row, mi >> 1));
    }
    const uint32_t bb = st + C::GBYTES;
    const int bk0 = (int)lds32(bb + (2 * q4) * 8), bk1 = (int)lds32(bb + (2 * q4 + 1) * 8), bk2 = (int)lds32(bb + (2 * q4 + 8) * 8), bk3 = (int)lds32(bb + (2 * q4 + 9) * 8);
    if constexpr (F32) {
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const uint32_t col = (8 * nt + g8) * 4;
        fv[nt][0] = __uint_as_float(lds32(st + (2 * q4) * 80 + col));
        fv[nt][1] = __uint_as_float(lds32(st + (2 * q4 + 1) * 80 + col));
        fv[nt][2] = __uint_as_float(lds32(st + (2 * q4 + 8) * 80 + col));
        fv[nt][3] = __uint_as_float(lds32(st + (2 * q4 + 9) * 80 + col));
      }
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        // hi + mid + lo = the fp32 value exactly (24 = 3 x 8 significand bits); pairs of rows pack as (lower row, upper row)
        float hi[4], mid[4], lo[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          hi[j] = round_bf16f(fv[nt][j]);
          const float r1 = fv[nt][j] - hi[j];
          mid[j] = round_bf16f(r1);
          lo[j] = r1 - mid[j];
        }
        bp[0][nt][0] = pack_bf16(hi[0], hi[1]);   bp[0][nt][1] = pack_bf16(hi[2], hi[3]);
        bp[1][nt][0] = pack_bf16(mid[0], mid[1]); bp[1][nt][1] = pack_bf16(mid[2], mid[3]);
        bp[2][nt][0] = pack_bf16(lo[0], lo[1]);   bp[2][nt][1] = pack_bf16(lo[2], lo[3]);
      }
    }
    __syncwarp();                                                  // every lane has read the stage: it is refilled below
    if (i + C::NSTG - 1 < nmine) issue(s0 + i + C::NSTG - 1, (int)((i + C::NSTG - 1) % C::NSTG));
    cp_async_commit();

    // ---- one-hot A fragments of the five bucket m-tiles, and the products
#pragma unroll
    for (int mt = 0; mt < 5; ++mt) {
      const int t0 = 16 * mt + g8, t1 = t0 + 8;
      uint32_t a[4];
      a[0] = ((bk0 == t0) ? 0x3F80u : 0u) | ((bk1 == t0) ? 0x3F800000u : 0u);
      a[1] = ((bk0 == t1) ? 0x3F80u : 0u) | ((bk1 == t1) ? 0x3F800000u : 0u);
      a[2] = ((bk2 == t0) ? 0x3F80u : 0u) | ((bk3 == t0) ? 0x3F800000u : 0u);
      a[3] = ((bk2 == t1) ? 0x3F80u : 0u) | ((bk3 == t1) ? 0x3F800000u : 0u);
      if (mt == 4) { a[1] |= ones; a[3] |= ones; }
      if constexpr (!F32) {
        mma16816(acc[mt][0], a, bfr[0], bfr[1]);
        mma16816(acc[mt][1], a, bfr[2], bfr[3]);
      } else {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          // b0 = rows (2 q4, 2 q4 + 1), b1 = rows (2 q4 + 8, 2 q4 + 9) of piece hi / mid / lo
          mma16816(acc[mt][nt], a, bp[0][nt][0], bp[0][nt][1]);
          mma16816(acc[mt][nt], a, bp[1][nt][0], bp[1][nt][1]);
          mma16816(acc[mt][nt], a, bp[2][nt][0], bp[2][nt][1]);
        }
      }
    }
  }
  cp_async_wait<0>();
  __syncthreads();                                                // the ring is dead: its shared memory holds the warps' tables
  float* tab = reinterpret_cast<float*>(smem_raw) + warp * RP_TAB;
#pragma unroll
  for (int mt = 0; mt < 5; ++mt)
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float* dst = tab + (16 * mt + g8 + 8 * h) * RP_W + 8 * nt + 2 * q4;
        dst[0] = acc[mt][nt][2 * h];
        dst[1] = acc[mt][nt][2 * h + 1];
      }
  __syncthreads();
  for (int idx = tid; idx < RP_TAB; idx += C::NTHR) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < C::NW; ++w) s += reinterpret_cast<const float*>(smem_raw)[w * RP_TAB + idx];
    p.partial[(size_t)blockIdx.x * RP_TAB + idx] = s;
  }
}

// the CTA partial tables -> grad_table [nbuckets, 16] and grad_bias [16], summed in CTA order
__global__ void relpos_reduce_kernel(const float* __restrict__ partial, float* __restrict__ table, float* __restrict__ bias, int nparts, int nbuckets) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= RP_TAB) return;
  float s = 0.f;
  for (int c = 0; c < nparts; ++c) s += partial[(size_t)c * RP_TAB + idx];
  const int row = idx / RP_W, ch = idx % RP_W;
  if (row < nbuckets) table[row * RP_W + ch] = s;
  if (row == RP_ROWS - 1) bias[ch] = s;
}

}  // namespace mp80
