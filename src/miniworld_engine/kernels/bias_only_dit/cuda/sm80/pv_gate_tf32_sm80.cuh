// pv_gate_tf32_sm80.cuh -- the TF32 twin of pv_gate_sm80.cuh: the attention core of the fp32 bias-only token DiT block on A100 / sm_80.
//
//   out[s L + i, h DH + d] = sigmoid(g[s L + i, h DH + d]) * sum_j P[h L + i, j] v[s L + j, h DH + d]        (without the gate: the product alone)
//
// Everything is fp32 in memory (P [NH L][L], v / g / out [S L][..] views); the products run on the tensor cores as mma.sync m16n8k8 TF32 (operands rounded to nearest tf32 in registers, as cuBLAS rounds its
// TF32 operands, fp32 accumulation).  Same schedule as the bf16 core: CTA = 128 queries of one head and a group of SG samples, 8 warps x 16 queries, an NST-stage cp.async ring of KC = 32 keys (a chunk is
// ONE P tile [128][32] fp32 = 128 B per row and the SG samples' v tiles [32][DH] fp32), so P is read once per group.  A comes from shared memory through ldmatrix (32-bit words are what its 8x8 b16 matrices
// carry: matrix 0 / 1 = rows 0-7 / 8-15 of the k8 block's first four words = a0 / a1, matrix 2 / 3 the last four = a2 / a3); B (k8 x n8, k = key, n = channel) is two 32-bit loads per MMA (ldmatrix.trans moves
// 16-bit elements, not words).  Rows are padded (P 144 B, v (DH + 8) words) so that both are conflict-free without a swizzle.
#pragma once
#include "sm80_common.cuh"

namespace bo80 {

DEVI uint32_t tf32_rna(float x) { uint32_t r; asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(x)); return r; }
DEVI void mma1688(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// MT_ = m16 tiles per warp (1: 8 warps x 16 queries; 2: 4 warps x 32 queries, each B fragment loaded once for two A fragments)
template <int DH_, int NST_, int SG_, int MT_ = 1>
struct PvTCfg {
  static constexpr int DH = DH_, NT = DH_ / 8, MT = MT_, NTHR = 256 / MT_, QT = 128, KC = 32, SG = SG_, NST = NST_;
  static constexpr int PROW = KC * 4 + 16;                 // 144 B: 9 granules
  static constexpr int VSTR = DH_ + 8;                     // words per v row: 40 / 56 / 72 (bank = 8 q + g: conflict-free)
  static constexpr int VROW = VSTR * 4;
  static constexpr int PBYTES = QT * PROW, VBYTES = KC * VROW, STAGE = PBYTES + SG_ * VBYTES;
  static constexpr int SMEM = NST_ * STAGE;
};

struct PvTParams {
  const float* P;               // [NH L][L]
  const float* v;               // [S L][..], row stride ldv, head h at columns h DH
  const float* g;               // gate, or nullptr
  float* out;                   // [S L][..], row stride ldo
  int L, nh, S;
  long long ldv, ldg, ldo;
};

template <class G>
DEVI void pvt_load_chunk(const PvTParams& p, uint32_t stage, int c, int q0, int h, int s0, int ns, int tid) {
  const int k0 = c * G::KC;
#pragma unroll
  for (int r = 0; r < 1024 / G::NTHR; ++r) {                            // P tile: 128 rows of 32 keys (8 granules per row)
    const int i = tid + G::NTHR * r, row = i >> 3, gr = i & 7;
    cp_async16(stage + row * G::PROW + gr * 16, p.P + ((size_t)h * p.L + q0 + row) * p.L + k0 + gr * 4);
  }
  constexpr int GPR = G::DH / 4;                           // granules per v row
  for (int i = tid; i < ns * G::KC * GPR; i += G::NTHR) {
    const int smp = i / (G::KC * GPR), rem = i - smp * (G::KC * GPR), row = rem / GPR, gr = rem - row * GPR;
    cp_async16(stage + G::PBYTES + smp * G::VBYTES + row * G::VROW + gr * 16, p.v + ((size_t)(s0 + smp) * p.L + k0 + row) * p.ldv + (size_t)h * G::DH + gr * 4);
  }
}

template <class G>
__global__ void __launch_bounds__(G::NTHR, 1) pv_gate_tf32_kernel(const PvTParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int q0 = blockIdx.x * G::QT, h = blockIdx.y, s0 = blockIdx.z * G::SG;
  const int ns = min(G::SG, p.S - s0);
  const int nch = p.L / G::KC;

  float o[G::MT][G::SG][G::NT][4];
#pragma unroll
  for (int m = 0; m < G::MT; ++m)
#pragma unroll
  for (int i = 0; i < G::SG; ++i)
#pragma unroll
    for (int t = 0; t < G::NT; ++t) { o[m][i][t][0] = o[m][i][t][1] = o[m][i][t][2] = o[m][i][t][3] = 0.f; }

#pragma unroll
  for (int c0 = 0; c0 < G::NST - 1; ++c0) {
    if (c0 < nch) pvt_load_chunk<G>(p, sb + c0 * G::STAGE, c0, q0, h, s0, ns, tid);
    cp_async_commit();
  }
  for (int c = 0; c < nch; ++c) {
    cp_async_wait<G::NST - 2>();
    __syncthreads();
    if (c + G::NST - 1 < nch) pvt_load_chunk<G>(p, sb + ((c + G::NST - 1) % G::NST) * G::STAGE, c + G::NST - 1, q0, h, s0, ns, tid);
    cp_async_commit();
    const uint32_t pst = sb + (c % G::NST) * G::STAGE, vst = pst + G::PBYTES;
#pragma unroll
    for (int ks = 0; ks < G::KC / 8; ++ks) {
      uint32_t a[G::MT][4];
#pragma unroll
      for (int m = 0; m < G::MT; ++m) {
        ldsm_x4(a[m], pst + (16 * (warp * G::MT + m) + (lane & 7) + ((lane >> 3) & 1) * 8) * G::PROW + (lane >> 4) * 16 + ks * 32);
#pragma unroll
        for (int k = 0; k < 4; ++k) a[m][k] = tf32_rna(__uint_as_float(a[m][k]));
      }
#pragma unroll
      for (int i = 0; i < G::SG; ++i) {
        if (i < ns) {
          const uint32_t vb = vst + i * G::VBYTES + (ks * 8 + q4) * G::VROW + g8 * 4;
#pragma unroll
          for (int nt = 0; nt < G::NT; ++nt) {
            const uint32_t b0 = tf32_rna(__uint_as_float(lds32(vb + nt * 32))), b1 = tf32_rna(__uint_as_float(lds32(vb + nt * 32 + 4 * G::VROW)));
#pragma unroll
            for (int m = 0; m < G::MT; ++m) mma1688(o[m][i][nt], a[m], b0, b1);
          }
        }
      }
    }
  }

  // ---- epilogue: the gate, fp32 out
#pragma unroll
  for (int mt = 0; mt < G::MT; ++mt) {
    const int r0 = q0 + 16 * (warp * G::MT + mt) + g8, r1 = r0 + 8;
#pragma unroll
    for (int i = 0; i < G::SG; ++i) {
      if (i < ns) {
        const size_t ro0 = (size_t)(s0 + i) * p.L + r0, ro1 = (size_t)(s0 + i) * p.L + r1;
#pragma unroll
        for (int nt = 0; nt < G::NT; ++nt) {
          const int col = 8 * nt + 2 * q4;
          float x0 = o[mt][i][nt][0], x1 = o[mt][i][nt][1], x2 = o[mt][i][nt][2], x3 = o[mt][i][nt][3];
          if (p.g != nullptr) {
            const float2 u0 = *reinterpret_cast<const float2*>(p.g + ro0 * p.ldg + (size_t)h * G::DH + col), u1 = *reinterpret_cast<const float2*>(p.g + ro1 * p.ldg + (size_t)h * G::DH + col);
            x0 *= sigmoidf(u0.x); x1 *= sigmoidf(u0.y);
            x2 *= sigmoidf(u1.x); x3 *= sigmoidf(u1.y);
          }
          *reinterpret_cast<float2*>(p.out + ro0 * p.ldo + (size_t)h * G::DH + col) = make_float2(x0, x1);
          *reinterpret_cast<float2*>(p.out + ro1 * p.ldo + (size_t)h * G::DH + col) = make_float2(x2, x3);
        }
      }
    }
  }
}

}  // namespace bo80
