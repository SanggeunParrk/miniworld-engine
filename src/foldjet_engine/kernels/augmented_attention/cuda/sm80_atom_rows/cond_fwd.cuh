// cond_fwd.cuh -- the conditioning projections of the AF3 local atom DiT block on A100 / sm_80 (per row r of M = A N; c bf16 [M, 128]; the bf16 module's rounding points):
//
//   cn_k = rn(LayerNorm_{g_k}(c))     (weight only, eps 1e-5; k = 0 the attention's AdaLN, 1 its key / value AdaLN, 2 the transition's AdaLN)
//   mod  = [ s1 | bi1 | skv | bikv | s2 | bi2 | so | st ]  bf16 [M, 1024]:
//     s1 = rn(sigmoid(rn(cn_0 Wsc1^T + bsc1)))   bi1 = rn(cn_0 Wbi1^T)     skv = rn(sigmoid(rn(cn_1 Wsckv^T + bsckv)))   bikv = rn(cn_1 Wbikv^T)
//     s2 = rn(sigmoid(rn(cn_2 Wsc2^T + bsc2)))   bi2 = rn(cn_2 Wbi2^T)     so = rn(sigmoid(rn(c Wos^T + bos)))   st = rn(sigmoid(rn(c Wts^T + bts)))
//   (the module uses the scales and gates only through their sigmoids, so they leave as rn(sigmoid): the same rounding points, and what autograd's sigmoid backward reads.)
//
// A CTA of NW warps takes tiles of 32 NW rows (a warp: 32 rows = two row blocks of 16); the packed weight W [1024][128] (the eight [128][128] matrices in the order of the mod columns)
// streams through a cp.async ring in chunks of 64 output channels, shared by the warps of the CTA. Every four chunks the A operand changes (cn_0, cn_1, cn_2, c): the warp rebuilds its
// fragments from its raw rows (staged in shared memory, 8 KB a warp, requested a tile ahead) with the row statistics it computed at the start of the tile.
#pragma once
#include "rows_layout.cuh"

namespace lr80 {

constexpr int CF_N = 1024, CF_CH = 64, CF_NCH = CF_N / CF_CH;

struct CondFwdParams {
  const __nv_bfloat16* c;                            // [M][128]
  const __nv_bfloat16* w;                            // [1024][128]
  const float* bias;                                 // [1024]
  const float* gamma;                                // [3][128]
  __nv_bfloat16* mod;                                // [M][1024]
  int M;
  float eps;
  int cross;                                         // 0: the key / value AdaLN's two weight blocks (mod columns 256 .. 511) are skipped
};

template <int NW_, int MINB_, int NSTAGE_, int MT_ = 2>
struct CondCfg {
  static constexpr int NW = NW_, NTHR = 32 * NW_, MT = MT_, RW = 16 * MT_, ROWS = RW * NW_, MINB = MINB_, NSTAGE = NSTAGE_, W_BYTES = CF_CH * 256;     // RW rows a warp
  static constexpr int O_RAW = NSTAGE * W_BYTES, RAW_BYTES = RW * 256, O_G = O_RAW + NW * RAW_BYTES, O_B = O_G + 3 * 128 * 4, SMEM = O_B + CF_N * 4;
};

DEVI int cf_chunk(int j, int cross) { return (!cross && j >= 4) ? j + 4 : j; }          // the weight chunk of the j-th chunk of a tile

template <class G>
DEVI void cf_load_w(const CondFwdParams& p, uint32_t stage, int j, int tid) {        // packed rows 64 j .. 64 j + 63 in the f1 order
#pragma unroll
  for (int i = tid; i < CF_CH * 16; i += G::NTHR) {
    const int r = i >> 4, ch = i & 15;
    cp_async16(stage + qf_woff(r, ch), p.w + (size_t)qf_channel(CF_CH * j + r) * 128 + ch * 8);
  }
}
template <class G>
DEVI void cf_load_rows(const CondFwdParams& p, uint32_t raw, int t0, int lane) {     // the warp's RW rows (zero past M) into its RAW_BYTES
#pragma unroll
  for (int i = lane; i < G::RW * 16; i += 32) {
    const int r = i >> 4, ch = i & 15, row = t0 + r;
    cp_async16(raw + qf_woff(r, ch), p.c + (size_t)(row < p.M ? row : 0) * 128 + ch * 8, row < p.M ? 16u : 0u);
  }
}

template <class G>
__global__ void __launch_bounds__(G::NTHR, G::MINB) cond_fwd_kernel(const CondFwdParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const float* sg = reinterpret_cast<const float*>(smem_raw + G::O_G);
  const float* sbias = reinterpret_cast<const float*>(smem_raw + G::O_B);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const uint32_t raw = sb + G::O_RAW + warp * G::RAW_BYTES;

  for (int i = tid; i < 3 * 128; i += G::NTHR) reinterpret_cast<float*>(smem_raw + G::O_G)[i] = p.gamma[i];
  for (int i = tid; i < CF_N; i += G::NTHR) reinterpret_cast<float*>(smem_raw + G::O_B)[i] = p.bias[i];

  const int ntile = (p.M + G::ROWS - 1) / G::ROWS;
  const int mine = (int)blockIdx.x < ntile ? (ntile - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  const int nch = p.cross ? CF_NCH : CF_NCH - 4;                           // chunks per tile
  const int total = mine * nch;
  if (mine > 0) cf_load_rows<G>(p, raw, (int)blockIdx.x * G::ROWS + warp * G::RW, lane);
  int iw = 0, ij = 0, ist = 0;                                             // the weight request stream: chunk, its index in the tile, the ring stage (counters, no divisions)
  auto next_w = [&]() {
    if (iw < total) cf_load_w<G>(p, sb + ist * G::W_BYTES, cf_chunk(ij, p.cross), tid);
    ++iw;
    if (++ij == nch) ij = 0;
    if (++ist == G::NSTAGE) ist = 0;
  };
#pragma unroll
  for (int s = 0; s < G::NSTAGE - 1; ++s) {
    next_w();
    cp_async_commit();
  }

  uint32_t wb[4];
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) wb[kb] = qf_woff(g8, 4 * kb + q4);

  float mrs[G::MT][2], rstd[G::MT][2];                                     // mean * rstd, rstd
  uint32_t af[G::MT][8][4];
  int jn = 0, tl = 0, cs = 0;                                              // the consumed chunk: its index in the tile, the tile, the ring stage
  for (int ci = 0; ci < total; ++ci) {
    cp_async_wait<G::NSTAGE - 2>();
    __syncthreads();                                                       // chunk ci (and, at a tile start, the raw rows) are in shared memory; the stage of chunk ci - 1 is free
    const int tile = (int)blockIdx.x + tl * (int)gridDim.x;
    const int j = cf_chunk(jn, p.cross);                                   // the weight chunk (0 .. 15)
    const int t0 = tile * G::ROWS + warp * G::RW;

    if (jn == 0) {                                                         // row statistics of the tile's raw rows
#pragma unroll
      for (int mt = 0; mt < G::MT; ++mt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int rl = 16 * mt + g8 + 8 * hh;
          float s = 0.f;
#pragma unroll
          for (int kb = 0; kb < 4; ++kb) {
            float x[8];
            unpack8(lds128(raw + qf_woff(rl, 4 * kb + q4)), x);
#pragma unroll
            for (int i = 0; i < 8; ++i) s += x[i];
          }
          const float m = quad_sum(s) * (1.f / 128.f);
          float v = 0.f;
#pragma unroll
          for (int kb = 0; kb < 4; ++kb) {
            float x[8];
            unpack8(lds128(raw + qf_woff(rl, 4 * kb + q4)), x);
#pragma unroll
            for (int i = 0; i < 8; ++i) { const float d = x[i] - m; v = fmaf(d, d, v); }
          }
          const float rs = 1.f / sqrtf(quad_sum(v) * (1.f / 128.f) + p.eps);
          rstd[mt][hh] = rs;
          mrs[mt][hh] = m * rs;
        }
    }
    if ((j & 3) == 0) {                                                    // the A operand of this group of four chunks: cn_0, cn_1, cn_2, raw c
      const int var = j >> 2;
#pragma unroll
      for (int mt = 0; mt < G::MT; ++mt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int rl = 16 * mt + g8 + 8 * hh;
#pragma unroll
          for (int kb = 0; kb < 4; ++kb) {
            uint4 u = lds128(raw + qf_woff(rl, 4 * kb + q4));
            if (var < 3) {
              float x[8];
              unpack8(u, x);
              const float* gp = sg + var * 128 + 32 * kb + 8 * q4;
              const float4 g0 = *reinterpret_cast<const float4*>(gp), g1 = *reinterpret_cast<const float4*>(gp + 4);
              const float gv[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
#pragma unroll
              for (int i = 0; i < 8; ++i) x[i] = fmaf(x[i], rstd[mt][hh], -mrs[mt][hh]) * gv[i];
              u = pack8(x);
            }
            put_a(af[mt], kb, hh, u);
          }
        }
    }
    {
      next_w();
      if (j == 12 && tl + 1 < mine) cf_load_rows<G>(p, raw, (tile + (int)gridDim.x) * G::ROWS + warp * G::RW, lane);   // the next tile's rows, the A operands of j == 12 are built
      cp_async_commit();
    }

    const uint32_t wst = sb + cs * G::W_BYTES;
    float acc[G::MT][8][4];
#pragma unroll
    for (int mt = 0; mt < G::MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) { acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f; }
#pragma unroll
    for (int kb = 0; kb < 4; ++kb) {
      uint4 b[8];
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) b[nt] = lds128_ro(wst + nt * 8 * 256 + wb[kb]);
#pragma unroll
      for (int half = 0; half < 2; ++half)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
          for (int mt = 0; mt < G::MT; ++mt) mma16816(acc[mt][nt], af[mt][2 * kb + half], half ? b[nt].z : b[nt].x, half ? b[nt].w : b[nt].y);
    }

    const bool sigmoid_cols = (j >> 2) == 3 || (j & 3) < 2;                // scales and gates: sigmoid; the shifts: plain
    const bool has_bias = sigmoid_cols;
#pragma unroll
    for (int og = 0; og < 2; ++og) {
      const int col = CF_CH * j + 32 * og + 8 * q4;
      float bv[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      if (has_bias) {
        const float4 b0 = *reinterpret_cast<const float4*>(sbias + col), b1 = *reinterpret_cast<const float4*>(sbias + col + 4);
        bv[0] = b0.x; bv[1] = b0.y; bv[2] = b0.z; bv[3] = b0.w; bv[4] = b1.x; bv[5] = b1.y; bv[6] = b1.z; bv[7] = b1.w;
      }
#pragma unroll
      for (int mt = 0; mt < G::MT; ++mt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          uint32_t o[4];
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) {
            const float v0 = acc[mt][4 * og + jj][2 * hh], v1 = acc[mt][4 * og + jj][2 * hh + 1];
            if (sigmoid_cols) {
              const uint32_t pr = pk2(v0 + bv[2 * jj], v1 + bv[2 * jj + 1]);            // rn(acc + bias)
              o[jj] = pk2(sigm(bf16lo(pr)), sigm(bf16hi(pr)));
            } else {
              o[jj] = pk2(v0, v1);
            }
          }
          const int r = t0 + 16 * mt + g8 + 8 * hh;
          if (r < p.M) stg128(p.mod + (size_t)r * CF_N + col, make_uint4(o[0], o[1], o[2], o[3]));
        }
    }
    if (++jn == nch) { jn = 0; ++tl; }
    if (++cs == G::NSTAGE) cs = 0;
  }
}

}  // namespace lr80
