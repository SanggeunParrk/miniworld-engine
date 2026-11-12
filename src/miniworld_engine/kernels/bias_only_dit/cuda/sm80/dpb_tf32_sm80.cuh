// dpb_tf32_sm80.cuh -- the TF32 twin of dpb_sm80.cuh: the bias gradient of the fp32 bias-only token DiT on A100 / sm_80,
//
//   dbias_h[i, j] = P_h[i, j] (sum_a sum_d do[a, i, h, d] v[a, j, h, d] - D_h[i]),   D_h[i] = sum_a dd[a, h, i]
//
// All operands fp32 (do / v [A L][..] token-major views, P / dbias [NH L][L]); the products on the tensor cores as mma.sync m16n8k8 TF32 (operands rounded to nearest tf32, fp32 accumulation, a fixed
// order: the sample loop is the K loop, so a rerun is bit-identical).  Same CTA as the bf16 kernel: (head, 128 queries, 128 keys), 8 warps as 4 x 2, a warp tile of 32 x 64; a stage of the cp.async
// ring is one sample's do tile [128][DH] and v tile [128][DH] fp32 (rows padded to an odd number of 16-byte granules).  Both operands are K-contiguous, so ldmatrix (32-bit words as 8x8 b16 matrices)
// gives the A fragments (do) and the B fragments (v: row = key = n, words = k) without a transpose.
#pragma once
#include "pv_gate_tf32_sm80.cuh"

namespace bo80 {

template <int DH_, int NST_, int MINB_>
struct DpbTCfg {
  static constexpr int DH = DH_, NST = NST_, MINB = MINB_, NTHR = 256, BM = 128, BN = 128, KS = DH_ / 8;
#ifdef MWE_SM120
  // RTX 5090 (99 KB a block): head width 48 takes unpadded rows (192 B) with the granules rotated by the row (granule g of row r at (g + r) % 12: the 8 rows of an
  // ldmatrix fall in 8 distinct 16-B bank groups) -- 98816 B in place of the padded 107008
  static constexpr bool ROT = DH_ == 48;
#else
  static constexpr bool ROT = false;
#endif
  static constexpr int ROW = ROT ? DH_ * 4 : DH_ * 4 + 16;      // 144 / 208 / 272 B: 9 / 13 / 17 granules (sm_120a, 48: 192 B rotated)
  static constexpr int ABYTES = BM * ROW, BBYTES = BN * ROW;
  static constexpr int STAGE = ABYTES + BBYTES;
  static constexpr int SMEM = NST_ * STAGE + BM * 4;
#ifdef MWE_SM120
  static_assert(SMEM <= 101376, "dpb_tf32: the RTX 5090 has 101376 B of shared memory per block");
#endif
  static DEVI uint32_t off(int row, int gr) { return (uint32_t)(row * ROW + (ROT ? (gr + row) % (DH_ / 4) : gr) * 16); }   // byte offset of granule gr of row
};

struct DpbTParams {
  const float* dout;            // [A L][..], row stride lddo
  const float* v;               // [A L][..], row stride ldv
  const float* P;               // [NH L][L]
  const float* dd;              // [A][NH][L]
  float* dbias;                 // [NH L][L]
  int L, nh, A;
  long long lddo, ldv;
};

template <class G>
DEVI void dpbt_load_stage(const DpbTParams& p, uint32_t stage, int a, int i0, int j0, int h, int tid) {
  constexpr int GPR = G::DH / 4;                                   // 16-byte granules per row
  const size_t row0 = (size_t)a * p.L;
  for (int i = tid; i < G::BM * GPR; i += G::NTHR) {
    const int row = i / GPR, gr = i - row * GPR;
    cp_async16(stage + G::off(row, gr), p.dout + (row0 + i0 + row) * p.lddo + (size_t)h * G::DH + gr * 4);
  }
  for (int i = tid; i < G::BN * GPR; i += G::NTHR) {
    const int row = i / GPR, gr = i - row * GPR;
    cp_async16(stage + G::ABYTES + G::off(row, gr), p.v + (row0 + j0 + row) * p.ldv + (size_t)h * G::DH + gr * 4);
  }
}

template <class G>
__global__ void __launch_bounds__(G::NTHR, G::MINB) dpb_tf32_kernel(const DpbTParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  float* Ds = reinterpret_cast<float*>(smem_raw + G::NST * G::STAGE);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int wm = warp & 3, wn = warp >> 2;
  const int j0 = blockIdx.x * G::BN, i0 = blockIdx.y * G::BM, h = blockIdx.z;
  const int nch = p.A;

  float acc[2][8][4];
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int t = 0; t < 8; ++t) { acc[m][t][0] = acc[m][t][1] = acc[m][t][2] = acc[m][t][3] = 0.f; }

#pragma unroll
  for (int c0 = 0; c0 < G::NST - 1; ++c0) {
    if (c0 < nch) dpbt_load_stage<G>(p, sb + c0 * G::STAGE, c0, i0, j0, h, tid);
    cp_async_commit();
  }
  if (tid < G::BM) {                                               // D[i] = sum_a dd[a, h, i]
    float s = 0.f;
    for (int a = 0; a < p.A; ++a) s += __ldg(p.dd + ((size_t)a * p.nh + h) * p.L + i0 + tid);
    Ds[tid] = s;
  }

  for (int c = 0; c < nch; ++c) {
    cp_async_wait<G::NST - 2>();
    __syncthreads();
    const int cn = c + G::NST - 1;
    if (cn < nch) dpbt_load_stage<G>(p, sb + (cn % G::NST) * G::STAGE, cn, i0, j0, h, tid);
    cp_async_commit();
    const uint32_t sa = sb + (c % G::NST) * G::STAGE, sbb = sa + G::ABYTES;
#pragma unroll
    for (int ks = 0; ks < G::KS; ++ks) {
      uint32_t a[2][4];
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        ldsm_x4(a[m], sa + G::off(32 * wm + 16 * m + (lane & 7) + ((lane >> 3) & 1) * 8, (lane >> 4) + ks * 2));
#pragma unroll
        for (int k = 0; k < 4; ++k) a[m][k] = tf32_rna(__uint_as_float(a[m][k]));
      }
#pragma unroll
      for (int nn = 0; nn < 4; ++nn) {
        uint32_t b[4];
        const int mi = lane >> 3;
        ldsm_x4(b, sbb + G::off(64 * wn + 16 * nn + (mi >> 1) * 8 + (lane & 7), ks * 2 + (mi & 1)));
#pragma unroll
        for (int k = 0; k < 4; ++k) b[k] = tf32_rna(__uint_as_float(b[k]));
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          mma1688(acc[m][2 * nn], a[m], b[0], b[1]);
          mma1688(acc[m][2 * nn + 1], a[m], b[2], b[3]);
        }
      }
    }
  }

  // ---- epilogue: dbias = P (dP - D)
#pragma unroll
  for (int m = 0; m < 2; ++m) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int lr = 32 * wm + 16 * m + g8 + 8 * half;
      const float dsum = Ds[lr];
      const size_t base = ((size_t)h * p.L + i0 + lr) * p.L + j0 + 64 * wn + 2 * q4;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const float2 pv = __ldg(reinterpret_cast<const float2*>(p.P + base + 8 * nt));
        *reinterpret_cast<float2*>(p.dbias + base + 8 * nt) = make_float2(pv.x * (acc[m][nt][2 * half] - dsum), pv.y * (acc[m][nt][2 * half + 1] - dsum));
      }
    }
  }
}

}  // namespace bo80
