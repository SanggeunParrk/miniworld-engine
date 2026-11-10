// dpb_sm80.cuh -- the bias-only attention's bias gradient on A100 / sm_80 (the training backward; the twin of ``dpb_sm100.cu``):
//
//   dP_h[i, j]    = sum_a sum_d do[a, i, h, d] v[a, j, h, d]          (the attention weights P are shared by the A samples, so the samples are the contraction's outer loop)
//   dbias_h[i, j] = P_h[i, j] (dP_h[i, j] - D_h[i]),   D_h[i] = sum_a dd[a, h, i]      (dd [A, NH, L] fp32: the per-sample row term of the backward, sum_d do o)
//
// do / v are [A L][..] bf16 token-major views (head h at columns DH h, any row stride); P, dbias [NH L][L] bf16 (head-major, query rows).  A masked key has P = 0 and so dbias = 0.
//
// One CTA = (head, 128 queries, 128 keys): 8 warps as 4 (queries) x 2 (keys), a warp tile of 32 x 64 (2 x 8 mma tiles, 64 fp32 accumulators).  The K loop is the sample loop: a stage of the
// NST-stage cp.async ring holds SG samples' do tile [128][DH] and v tile [128][DH] (rows padded to an odd number of 16-byte granules, so every ldmatrix is conflict-free without a swizzle);
// both operands are K-contiguous, so ldmatrix WITHOUT .trans yields the A fragments (do) and the B fragments (v, rows = keys = the n index) of mma.m16n8k16.  The epilogue reads P in
// the accumulator layout (a 4-byte pair per fragment row) and writes dbias the same way; D comes from a prologue pass over dd (shared memory).
#pragma once
#include "sm80_common.cuh"

namespace bo80 {

template <int DH_, int NST_, int SG_, int MINB_>
struct DpbCfg {
  static constexpr int DH = DH_, NST = NST_, SG = SG_, MINB = MINB_, NTHR = 256, BM = 128, BN = 128, KS = DH_ / 16;
  static constexpr int ROW = DH_ * 2 + 16;                       // 80 / 112 / 144 B: 5 / 7 / 9 granules
  static constexpr int ABYTES = BM * ROW, BBYTES = BN * ROW;
  static constexpr int SAMPLE = ABYTES + BBYTES, STAGE = SG_ * SAMPLE;
  static constexpr int SMEM = NST_ * STAGE + BM * 4;             // the ring + D[128] fp32
};

struct DpbParams {
  const __nv_bfloat16* dout;    // [A L][..], row stride lddo
  const __nv_bfloat16* v;       // [A L][..], row stride ldv
  const __nv_bfloat16* P;       // [NH L][L]
  const float* dd;              // [A][NH][L]
  __nv_bfloat16* dbias;         // [NH L][L]
  int L, nh, A;
  long long lddo, ldv;
  long long hsd, hsv;           // element offset of head h in do / v: DH, or a plane stride (see pv_gate_sm80.cuh)
};

template <class G>
DEVI void dpb_load_stage(const DpbParams& p, uint32_t stage, int c, int i0, int j0, int h, int ns, int tid) {
  constexpr int GPR = G::DH / 8;                                   // 16-byte granules per row
  const int a0 = c * G::SG;
  for (int s = 0; s < ns; ++s) {
    const size_t a = (size_t)(a0 + s) * p.L;
    const uint32_t sa = stage + s * G::SAMPLE, sb = sa + G::ABYTES;
    for (int i = tid; i < G::BM * GPR; i += G::NTHR) {
      const int row = i / GPR, gr = i - row * GPR;
      cp_async16(sa + row * G::ROW + gr * 16, p.dout + (a + i0 + row) * p.lddo + (size_t)h * p.hsd + gr * 8);
    }
    for (int i = tid; i < G::BN * GPR; i += G::NTHR) {
      const int row = i / GPR, gr = i - row * GPR;
      cp_async16(sb + row * G::ROW + gr * 16, p.v + (a + j0 + row) * p.ldv + (size_t)h * p.hsv + gr * 8);
    }
  }
}

template <class G>
__global__ void __launch_bounds__(G::NTHR, G::MINB) dpb_kernel(const DpbParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  float* Ds = reinterpret_cast<float*>(smem_raw + G::NST * G::STAGE);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int wm = warp & 3, wn = warp >> 2;
  const int j0 = blockIdx.x * G::BN, i0 = blockIdx.y * G::BM, h = blockIdx.z;
  const int nch = (p.A + G::SG - 1) / G::SG;

  float acc[2][8][4];
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int t = 0; t < 8; ++t) { acc[m][t][0] = acc[m][t][1] = acc[m][t][2] = acc[m][t][3] = 0.f; }

#pragma unroll
  for (int c0 = 0; c0 < G::NST - 1; ++c0) {
    if (c0 < nch) dpb_load_stage<G>(p, sb + c0 * G::STAGE, c0, i0, j0, h, min(G::SG, p.A - c0 * G::SG), tid);
    cp_async_commit();
  }
  // D[i] = sum_a dd[a, h, i] for the CTA's 128 query rows (visible to every warp after the loop's first barrier)
  if (tid < G::BM) {
    float s = 0.f;
    for (int a = 0; a < p.A; ++a) s += __ldg(p.dd + ((size_t)a * p.nh + h) * p.L + i0 + tid);
    Ds[tid] = s;
  }

  for (int c = 0; c < nch; ++c) {
    cp_async_wait<G::NST - 2>();
    __syncthreads();
    const int cn = c + G::NST - 1;
    if (cn < nch) dpb_load_stage<G>(p, sb + (cn % G::NST) * G::STAGE, cn, i0, j0, h, min(G::SG, p.A - cn * G::SG), tid);
    cp_async_commit();
    const int ns = min(G::SG, p.A - c * G::SG);
    const uint32_t st = sb + (c % G::NST) * G::STAGE;
    for (int s = 0; s < ns; ++s) {
      const uint32_t sa = st + s * G::SAMPLE, sbb = sa + G::ABYTES;
#pragma unroll
      for (int ks = 0; ks < G::KS; ++ks) {
        uint32_t a[2][4];
#pragma unroll
        for (int m = 0; m < 2; ++m) ldsm_x4(a[m], sa + (32 * wm + 16 * m + (lane & 15)) * G::ROW + (lane >> 4) * 16 + ks * 32);
#pragma unroll
        for (int nn = 0; nn < 4; ++nn) {
          uint32_t b[4];
          const int mi = lane >> 3;
          ldsm_x4(b, sbb + (64 * wn + 16 * nn + (mi >> 1) * 8 + (lane & 7)) * G::ROW + ks * 32 + (mi & 1) * 16);
#pragma unroll
          for (int m = 0; m < 2; ++m) {
            mma16816(acc[m][2 * nn], a[m], b[0], b[1]);
            mma16816(acc[m][2 * nn + 1], a[m], b[2], b[3]);
          }
        }
      }
    }
  }

  // ---- epilogue: dbias = P (dP - D), one rounding to bf16
#pragma unroll
  for (int m = 0; m < 2; ++m) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int lr = 32 * wm + 16 * m + g8 + 8 * half;
      const float dsum = Ds[lr];
      const size_t base = ((size_t)h * p.L + i0 + lr) * p.L + j0 + 64 * wn + 2 * q4;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const uint32_t pu = __ldg(reinterpret_cast<const unsigned int*>(p.P + base + 8 * nt));
        const float r0 = bf16lo(pu) * (acc[m][nt][2 * half] - dsum), r1 = bf16hi(pu) * (acc[m][nt][2 * half + 1] - dsum);
        stg32(p.dbias + base + 8 * nt, pack_bf16(r0, r1));
      }
    }
  }
}

}  // namespace bo80

namespace bo80 {

// dd[a][h][i] = sum_d dO[h-th plane][a L + i][d] o[...][d]: the row term D of the bias-only attention's backward, one thread per (token, plane); head planes [S L][DH] bf16 one after the other
// (``hs`` elements apart), S = A samples of L tokens each.
template <int DH>
__global__ void __launch_bounds__(128) delta_planes_kernel(const __nv_bfloat16* __restrict__ dO, const __nv_bfloat16* __restrict__ O, float* __restrict__ dd, int A, int L, int planes, long long hs) {
  const long long tokens = (long long)A * L;
  const long long t = (long long)blockIdx.x * 128 + threadIdx.x;
  const int h = blockIdx.y;
  if (t >= tokens) return;
  const uint4* d = reinterpret_cast<const uint4*>(dO + (long long)h * hs + t * DH);
  const uint4* o = reinterpret_cast<const uint4*>(O + (long long)h * hs + t * DH);
  float s = 0.f;
#pragma unroll
  for (int c = 0; c < DH / 8; ++c) {
    const uint4 u = __ldg(d + c), v = __ldg(o + c);
    const uint32_t a[4] = {u.x, u.y, u.z, u.w}, b[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      s = fmaf(bf16lo(a[k]), bf16lo(b[k]), s);
      s = fmaf(bf16hi(a[k]), bf16hi(b[k]), s);
    }
  }
  const long long a_i = t / L, i = t - a_i * L;
  dd[(a_i * planes + h) * L + i] = s;
}

}  // namespace bo80
