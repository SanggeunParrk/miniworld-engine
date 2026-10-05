// pv_gate_sm80.cuh -- the attention core of the bias-only token DiT on A100 / sm_80 (mma.sync, ldmatrix, cp.async):
//
//   out[s L + i, h DH + d] = sigmoid(g[s L + i, h DH + d]) * sum_j P[h L + i, j] v[s L + j, h DH + d]        (without the gate: the product alone)
//
// P [NH L][L] bf16 is one block's attention weights, softmax(pair bias): the same for every sample, so it is the A operand and each sample's v the B operand of one GEMM per head.  v / g / out are [S L][W]
// bf16 views (W = NH DH) with any row stride: in the step v | g are the column halves of the v | g GEMM output.  The product accumulates in fp32 over the bf16 operands as given, the gate and the
// result's single rounding to bf16 are the epilogue (the reference ``kernels/bias_only_dit/reference.pv_gate``).
//
// One CTA = 128 queries of one head and a group of SG samples: 8 warps x 16 queries.  The keys stream through an NST-stage cp.async ring of KC = 64 keys, a chunk being ONE P tile [128][64] and the SG
// samples' v tiles [64][DH]: P is read once per group, not once per sample, and each warp issues SG x DH / 8 MMAs per P fragment.  Rows are padded to an odd number of 16-byte granules (P 144 B, v 80 /
// 112 / 144 B), so every ldmatrix is conflict-free without a swizzle.  P comes out of shared memory through ldmatrix as the A fragments, v through ldmatrix.trans as the B fragments of DH / 8 n8 tiles.
// A single CTA per SM is request-bound (16-byte cp.async granules, ~18 GB/s per SM measured), so the grouping, not a deeper ring, is what shortens the core: see the table in cuda/sm80/__init__.py.
#pragma once
#include "sm80_common.cuh"

namespace bo80 {

template <int DH_, int NST_, int SG_>
struct PvCfg {
  static constexpr int DH = DH_, NT = DH_ / 8, NTHR = 256, QT = 128, KC = 64, SG = SG_, NST = NST_;
  static constexpr int PROW = KC * 2 + 16;                 // 144 B: 9 granules
  static constexpr int VROW = DH_ * 2 + 16;                // 80 / 112 / 144 B: 5 / 7 / 9 granules
  static constexpr int PBYTES = QT * PROW, VBYTES = KC * VROW, STAGE = PBYTES + SG_ * VBYTES;
  static constexpr int SMEM = NST_ * STAGE;
};

struct PvParams {
  const __nv_bfloat16* P;       // [NH L][L]
  const __nv_bfloat16* v;       // [S L][..], row stride ldv, head h at columns h DH
  const __nv_bfloat16* g;       // gate, or nullptr
  __nv_bfloat16* out;           // [S L][..], row stride ldo
  int L, nh, S;
  long long ldv, ldg, ldo;
  long long hsv, hsg, hso;      // element offset of head h in v / g / out: DH (head h at columns DH h of token rows), or a plane stride (head planes [S L][DH] one after the other: ldv = DH)
};

template <class G>
DEVI void pv_load_chunk(const PvParams& p, uint32_t stage, int c, int q0, int h, int s0, int ns, int tid) {
  const int k0 = c * G::KC;
  // P tile: 128 rows of 64 keys (8 granules per row)
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int i = tid + G::NTHR * r, row = i >> 3, gr = i & 7;
    cp_async16(stage + row * G::PROW + gr * 16, p.P + ((size_t)h * p.L + q0 + row) * p.L + k0 + gr * 8);
  }
  // v tiles of the group's samples: 64 keys of DH channels (DH / 8 granules of 16 bytes per row)
  constexpr int GPR = G::DH / 8;
  for (int i = tid; i < ns * G::KC * GPR; i += G::NTHR) {
    const int smp = i / (G::KC * GPR), rem = i - smp * (G::KC * GPR), row = rem / GPR, gr = rem - row * GPR;
    cp_async16(stage + G::PBYTES + smp * G::VBYTES + row * G::VROW + gr * 16, p.v + ((size_t)(s0 + smp) * p.L + k0 + row) * p.ldv + (size_t)h * p.hsv + gr * 8);
  }
}

template <class G>
__global__ void __launch_bounds__(G::NTHR, 1) pv_gate_kernel(const PvParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int q0 = blockIdx.x * G::QT, h = blockIdx.y, s0 = blockIdx.z * G::SG;
  const int ns = min(G::SG, p.S - s0);                      // the samples of this group (the last group may be short)
  const int nch = p.L / G::KC;

  float o[G::SG][G::NT][4];
#pragma unroll
  for (int i = 0; i < G::SG; ++i)
#pragma unroll
    for (int t = 0; t < G::NT; ++t) { o[i][t][0] = o[i][t][1] = o[i][t][2] = o[i][t][3] = 0.f; }

  // NST-stage ring, one barrier per chunk: chunks 0 .. NST - 2 are in flight before the first product; iteration c waits for chunk c, passes the barrier (every warp is done with chunk c - 1, whose stage
  // chunk c + NST - 1 now takes) and issues that load; a group is committed every iteration, empty at the end, so the wait count is the same throughout
#pragma unroll
  for (int c0 = 0; c0 < G::NST - 1; ++c0) {
    if (c0 < nch) pv_load_chunk<G>(p, sb + c0 * G::STAGE, c0, q0, h, s0, ns, tid);
    cp_async_commit();
  }
  for (int c = 0; c < nch; ++c) {
    cp_async_wait<G::NST - 2>();
    __syncthreads();
    if (c + G::NST - 1 < nch) pv_load_chunk<G>(p, sb + ((c + G::NST - 1) % G::NST) * G::STAGE, c + G::NST - 1, q0, h, s0, ns, tid);
    cp_async_commit();
    const uint32_t pst = sb + (c % G::NST) * G::STAGE, vst = pst + G::PBYTES;
#pragma unroll
    for (int ks = 0; ks < G::KC / 16; ++ks) {
      uint32_t a[4];
      ldsm_x4(a, pst + (16 * warp + (lane & 15)) * G::PROW + (lane >> 4) * 16 + ks * 32);
#pragma unroll
      for (int i = 0; i < G::SG; ++i) {
        if (i < ns) {
#pragma unroll
          for (int nn = 0; nn < G::NT / 2; ++nn) {
            uint32_t b[4];
            const int m = lane >> 3;
            ldsm_x4_t(b, vst + i * G::VBYTES + (16 * ks + ((m & 1) << 3) + (lane & 7)) * G::VROW + (nn * 16 + (m >> 1) * 8) * 2);
            mma16816(o[i][2 * nn], a, b[0], b[1]);
            mma16816(o[i][2 * nn + 1], a, b[2], b[3]);
          }
        }
      }
    }
  }

  // ---- epilogue: the gate and the one rounding to bf16
  const int r0 = q0 + 16 * warp + g8, r1 = r0 + 8;
#pragma unroll
  for (int i = 0; i < G::SG; ++i) {
    if (i < ns) {
      const size_t ro0 = (size_t)(s0 + i) * p.L + r0, ro1 = (size_t)(s0 + i) * p.L + r1;
#pragma unroll
      for (int nt = 0; nt < G::NT; ++nt) {
        const int col = 8 * nt + 2 * q4;                      // within the head; the head offset joins each pointer
        float x0 = o[i][nt][0], x1 = o[i][nt][1], x2 = o[i][nt][2], x3 = o[i][nt][3];
        if (p.g != nullptr) {
          const uint32_t u0 = *reinterpret_cast<const uint32_t*>(p.g + ro0 * p.ldg + (size_t)h * p.hsg + col), u1 = *reinterpret_cast<const uint32_t*>(p.g + ro1 * p.ldg + (size_t)h * p.hsg + col);
          x0 *= sigmoidf(bf16lo(u0)); x1 *= sigmoidf(bf16hi(u0));
          x2 *= sigmoidf(bf16lo(u1)); x3 *= sigmoidf(bf16hi(u1));
        }
        *reinterpret_cast<uint32_t*>(p.out + ro0 * p.ldo + (size_t)h * p.hso + col) = pack_bf16(x0, x1);
        *reinterpret_cast<uint32_t*>(p.out + ro1 * p.ldo + (size_t)h * p.hso + col) = pack_bf16(x2, x3);
      }
    }
  }
}

}  // namespace bo80
