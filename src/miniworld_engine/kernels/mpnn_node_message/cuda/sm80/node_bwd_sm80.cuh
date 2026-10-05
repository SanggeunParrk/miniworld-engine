// node_bwd_sm80.cuh -- the backward of the fused encoder node message on A100 (sm_80), but for the two weight-gradient GEMMs and the neighbour scatter (the integration finishes those):
//
//   replay (nothing was saved):  pre = bf16(bf16(q + bf16(E W1e^T)) + nb[idx])   act = bf16(gelu(pre))   hid = bf16(act W2^T + b2)
//   gh = bf16(g[group, o'] mask[n] / scale)   dh = bf16(float(gh) gelu'(hid))   db2 += dh                    (dh [rows, 128] and act [rows, 128] are written: dW2 = dh^T act, one cuBLAS GEMM)
//   dact = dh W2     dpre = bf16(dact gelu'(pre))      (written: dW1e = dpre^T E, one cuBLAS GEMM; and the neighbour gradient nb_grad[j] = sum of the dpre rows whose edge points at j: a segmented reduction)
//   dedge = bf16(dpre W1e)  (written: the gradient of the edge states)      dquery[g] = sum_k dpre[g, k]  (written, bf16)
//
// Per 16-row tile of a group, one warp, as the forward kernel (same pipeline: the edge tile is re-requested once GEMM 1 has consumed it, the gathered neighbour tile once it has been consumed).  The layouts chain
// without shuffles: GEMM 1 (natural, in two halves of 64 output channels: 32 accumulators live) -> pre / act in the C layout (n = g8 + 8 h, o = 8 j + 2 q4 (+1)) = the B layout of the transposed GEMM 2; its
// epilogue writes dh^T [o'][n] to a shared tile (STS.32), whose transposed ldmatrix are the A fragments of dact = dh W2 (also dh's rows for the output); dact comes out in the layout of pre, so gelu'(pre) needs
// no reload and dpre (packed) is directly the A fragments of dedge = dpre W1e.  The bias gradient and dquery are reduce-scatters over the quad / over the eight row lanes into persistent registers.  No atomics.
//
// The four [rows, 128] outputs (act, dh, dpre, dedge) do not leave from the fragment layout (a store there is 4 bytes per lane: 8 rows x 16 B per instruction, which throttles the load-store queue): each goes
// through a shared staging tile and leaves as 16-byte stores of whole 256-byte rows.  The staging tile is the neighbour tile for `act` (its next gather is requested right after) and the dh^T tile for the others
// (dh's rows are held in registers once, and written over the dh^T tile when its last transposed read is done).
#pragma once
#include "mpnn_common.cuh"

namespace mp80 {

struct NodeBwdParams {
  const bf* edge;         // [rows][128]
  const bf* query;        // [groups][128]
  const bf* nbt;          // [NN][128]
  const int64_t* idx;     // [rows]
  const bf* w1;           // [128][128]
  const bf* w2;           // [128][128]
  const bf* b2;           // [128]
  const void* mask;       // [rows] fp32 or bf16
  const float* gred;      // [groups][128]
  bf* dedge;              // [rows][128]
  bf* dpre;               // [rows][128]
  bf* dh;                 // [rows][128]
  bf* act;                // [rows][128]
  bf* dquery;             // [groups][128]
  float* db2_part;        // [ctas][128]
  int64_t groups;
  int K;
  float inv_scale;
};

struct NodeBwdCfg {
  static constexpr int NW = 8, NTHR = NW * 32, MINB = 1;
  static constexpr int W_BYTES = 2 * 128 * 256, BIAS_BYTES = 512, TILE = 16 * 256, WSTRIDE = 3 * TILE;   // the two weights | the bias | per warp: the edge tile | the gathered neighbour tile | dh^T [128 o'][16 n]
  static constexpr int SMEM = W_BYTES + BIAS_BYTES + NW * WSTRIDE;
};

// the dh^T tile: 128 rows (o') of 32 B (16 n), 16-B granule jn of row o' stored at granule jn ^ ((o' >> 2) & 1)
DEVI uint32_t ht_swz(uint32_t o, uint32_t jn) { return o * 32u + ((jn ^ ((o >> 2) & 1u)) << 4); }

// a staging tile [16 rows][256 B] (swizzled sw256) -> rows < nv, granules gran0 .. gran0 + W - 1 of the row-major global tensor `dst` (rows of 128 bf16): 16-byte stores
template <int W>
DEVI void flush_tile(uint32_t stg, bf* dst, int gran0, int nv, int lane) {
  constexpr int N = 16 * W / 32;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    const int c = lane + 32 * i, row = c / W, gran = gran0 + (c % W);
    if (row < nv) stg128(dst + row * 128 + gran * 8, lds128(stg + sw256(row, gran)));
  }
}

template <bool MASK_BF16>
__global__ void __launch_bounds__(NodeBwdCfg::NTHR, NodeBwdCfg::MINB) node_bwd_kernel(const NodeBwdParams p) {
  using C = NodeBwdCfg;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const uint32_t sw1 = sb, sw2 = sb + 128 * 256;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;
  const int K = p.K, ntl = (K + 15) >> 4;

  for (int i = tid; i < 2 * 2048; i += C::NTHR) {
    const int w = i >> 11, r = (i >> 4) & 127, g = i & 15;
    cp_async16(sb + w * 32768 + sw256(r, g), (w ? p.w2 : p.w1) + r * 128 + g * 8);
  }
  if (tid < 16) cp_async16(sb + C::W_BYTES + tid * 16, p.b2 + tid * 8);
  cp_async_commit();

  const int64_t total_warps = (int64_t)gridDim.x * C::NW;
  const int64_t gw = (int64_t)blockIdx.x * C::NW + warp;
  const int64_t ng = gw < p.groups ? (p.groups - gw + total_warps - 1) / total_warps : 0;
  const int64_t ntiles = ng * ntl;
  const uint32_t tbe = sb + C::W_BYTES + C::BIAS_BYTES + warp * C::WSTRIDE, tbn = tbe + C::TILE, tbh = tbn + C::TILE;

  const int mi = lane >> 3, x7 = lane & 7;
  const uint32_t ae = tbe + ((mi & 1) * 8 + x7) * 256;
  const int gae = mi >> 1;
  const int gb1 = mi & 1, rb1 = mi >> 1;
  const uint32_t aw2 = sw2 + ((mi & 1) * 8 + x7) * 256;
  const bf* sb2 = reinterpret_cast<const bf*>(smem_raw + C::W_BYTES);

  // tile k of this warp = tile t of its group number gi: k / ntl by an invariant-divisor multiply (ntl <= 8, k < 2^29: exact; a 64-bit division by a runtime value is a subroutine call)
  const uint32_t ntl_inv = ntl == 1 ? 0u : 0xFFFFFFFFu / (uint32_t)ntl + 1u;
  auto tile_info = [&](int64_t k, int64_t& row0, int& nv, int64_t& g, int& t) {
    const uint32_t ku = (uint32_t)k, giu = ntl == 1 ? ku : __umulhi(ku, ntl_inv);
    const int64_t gi = giu;
    t = (int)(ku - giu * (uint32_t)ntl);
    g = gw + gi * total_warps;
    row0 = g * K + 16 * t;
    nv = min(16, K - 16 * t);
  };
  auto load_idx = [&](int64_t k, int (&ix)[8]) {
    int64_t row0, g; int nv, t;
    tile_info(k, row0, nv, g, t);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int r = 2 * i + (lane >> 4);
      ix[i] = r < nv ? (int)p.idx[row0 + r] : 0;
    }
  };
  auto load_mask = [&](int64_t k, float (&mk)[4]) {
    int64_t row0, g; int nv, t;
    tile_info(k, row0, nv, g, t);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int n = 2 * q4 + (i & 1) + 8 * (i >> 1);
      float v = 0.f;
      if (n < nv) {
        if constexpr (MASK_BF16) v = __bfloat162float(reinterpret_cast<const bf*>(p.mask)[row0 + n]);
        else v = reinterpret_cast<const float*>(p.mask)[row0 + n];
      }
      mk[i] = v;
    }
  };
  auto issue_e = [&](int64_t k) {
    int64_t row0, g; int nv, t;
    tile_info(k, row0, nv, g, t);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = lane + 32 * i, r = c >> 4;
      const bool ok = r < nv;
      cp_async16(tbe + sw256(r, c & 15), p.edge + (ok ? (row0 + r) * 128 + (c & 15) * 8 : 0), ok ? 16 : 0);
    }
  };
  auto issue_nb = [&](int64_t k, const int (&ix)[8]) {
    int64_t row0, g; int nv, t;
    tile_info(k, row0, nv, g, t);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = lane + 32 * i, r = c >> 4;
      cp_async16(tbn + sw256(r, c & 15), p.nbt + (int64_t)ix[i] * 128 + (c & 15) * 8, r < nv ? 16 : 0);
    }
  };

  cp_async_wait<0>();
  __syncthreads();
  int ixn[8];
  float mkn[4];
  if (ntiles > 0) {
    load_idx(0, ixn);
    load_mask(0, mkn);
    issue_e(0);
    issue_nb(0, ixn);
  }
  cp_async_commit();

  float dbp[4] = {0.f, 0.f, 0.f, 0.f};                             // db2 reduce-scatter over the quad (as msg_bwd): lane q4 holds idx = base .. base + 3, idx = 2 m' + h <-> o' = 16 m' + g8 + 8 h
  float dqp[4] = {0.f, 0.f, 0.f, 0.f};                             // dquery reduce-scatter over the eight row lanes: see the end of a tile
  const bool hi2 = (q4 & 2) != 0, hi1 = (q4 & 1) != 0;
  const bool g4 = (g8 & 1) != 0, g8b = (g8 & 2) != 0, g16 = (g8 & 4) != 0;
  uint32_t qv[16];
  int tg = 0;
  for (int64_t k = 0; k < ntiles; ++k) {
    int64_t row0, g; int nv, t;
    tile_info(k, row0, nv, g, t);
    float mk[4] = {mkn[0], mkn[1], mkn[2], mkn[3]};
    if (tg == 0) {
#pragma unroll
      for (int j = 0; j < 16; ++j) qv[j] = *reinterpret_cast<const uint32_t*>(p.query + g * 128 + 8 * j + 2 * q4);
    }
    const float* gp = p.gred + g * 128;
    if (k + 1 < ntiles) { load_idx(k + 1, ixn); load_mask(k + 1, mkn); }
    cp_async_wait<0>();
    __syncwarp();

    // ---- replay GEMM 1 (two halves of 64 output channels) and its epilogue: pre = bf16(bf16(q + bf16(acc)) + nb), act = bf16(gelu(pre))
    uint32_t preP[16][2], actf[16][2];
    float gg16[8][2];                                              // g[16 m + g8 + 8 h] / scale: loaded early, used by the second GEMM's epilogue
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      float acc1[8][4];
#pragma unroll
      for (int j = 0; j < 8; ++j) { acc1[j][0] = acc1[j][1] = acc1[j][2] = acc1[j][3] = 0.f; }
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        uint32_t a[4];
        ldsm_x4(a, ae + (((2 * s + gae) ^ x7) << 4));
#pragma unroll
        for (int jp = 0; jp < 4; ++jp) {
          uint32_t b[4];
          ldsm_x4(b, sw1 + (8 * (2 * (4 * hh + jp) + rb1) + x7) * 256 + (((2 * s + gb1) ^ x7) << 4));
          mma16816(acc1[2 * jp], a, b[0], b[1]);
          mma16816(acc1[2 * jp + 1], a, b[2], b[3]);
        }
      }
      if (hh == 0) {
#pragma unroll
        for (int m = 0; m < 8; ++m)
#pragma unroll
          for (int h = 0; h < 2; ++h) gg16[m][h] = ldg_f32(gp + 16 * m + g8 + 8 * h) * p.inv_scale;
      } else {
        __syncwarp();                                              // the edge tile has been consumed
        if (k + 1 < ntiles) issue_e(k + 1);
        cp_async_commit();
      }
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) {
        const int j = 8 * hh + jj;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint32_t pr = pack_bf16(acc1[jj][2 * h], acc1[jj][2 * h + 1]);
          const uint32_t nbp = lds32(tbn + sw256(g8 + 8 * h, j) + 4 * q4);
          const uint32_t pre = node_pre(pr, qv[j], nbp);
          preP[j][h] = pre;
          actf[j][h] = pack_bf16(gelu_f(bf16lo(pre)), gelu_f(bf16hi(pre)));
        }
      }
    }
    // act leaves through the neighbour tile (every lane has read its own words of it: it overwrites the same words), whose next gather is requested right after
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
      for (int h = 0; h < 2; ++h) sts32(tbn + sw256(g8 + 8 * h, j) + 4 * q4, actf[j][h]);
    __syncwarp();
    flush_tile<16>(tbn, p.act + row0 * 128, 0, nv, lane);
    __syncwarp();
    if (k + 1 < ntiles) issue_nb(k + 1, ixn);
    cp_async_commit();

    // ---- GEMM 2 (transposed) and its epilogue: hid, dh^T -> the shared tile
    float vs[16];
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      float acc[2][4];
#pragma unroll
      for (int j = 0; j < 2; ++j) { acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f; }
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        uint32_t a[4];
        ldsm_x4(a, aw2 + m * 4096 + (((2 * s + (mi >> 1)) ^ x7) << 4));
        mma16816(acc[0], a, actf[2 * s][0], actf[2 * s + 1][0]);
        mma16816(acc[1], a, actf[2 * s][1], actf[2 * s + 1][1]);
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int o = 16 * m + g8 + 8 * h;
        const float b = __bfloat162float(sb2[o]), gg = gg16[m][h];
        float s2 = 0.f;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const uint32_t hid = pack_bf16(acc[j][2 * h] + b, acc[j][2 * h + 1] + b);
          const float gh0 = round_bf16f(gg * mk[2 * j]), gh1 = round_bf16f(gg * mk[2 * j + 1]);
          const uint32_t dh = pack_bf16(gh0 * gelu_grad_f(bf16lo(hid)), gh1 * gelu_grad_f(bf16hi(hid)));
          s2 += bf16lo(dh) + bf16hi(dh);
          sts32(tbh + ht_swz(o, j) + 4 * q4, dh);
        }
        vs[2 * m + h] = s2;
      }
    }
    {
      float w8[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float send = hi2 ? vs[i] : vs[i + 8], keep = hi2 ? vs[i + 8] : vs[i];
        w8[i] = keep + __shfl_xor_sync(0xffffffffu, send, 2);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float send = hi1 ? w8[i] : w8[i + 4], keep = hi1 ? w8[i + 4] : w8[i];
        dbp[i] += keep + __shfl_xor_sync(0xffffffffu, send, 1);
      }
    }
    __syncwarp();

    // ---- dact = dh W2 in two halves of 64 channels -> dpre = bf16(dact gelu'(pre)): the A fragments of dedge, dpre[j][h] = the pair (n = g8 + 8 h, o = 8 j + 2 q4 (+1))
    uint32_t dpre[16][2];
    uint32_t ad[8][4];                                             // dh as A fragments (rows n, k = o'): read once, they are also dh's rows for the output
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      float accx[8][4];
#pragma unroll
      for (int j = 0; j < 8; ++j) { accx[j][0] = accx[j][1] = accx[j][2] = accx[j][3] = 0.f; }
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        if (half == 0) ldsm_x4_t(ad[s], tbh + ht_swz(16 * s + 8 * (mi >> 1) + x7, mi & 1));
#pragma unroll
        for (int jp = 0; jp < 4; ++jp) {
          uint32_t bw[4];
          ldsm_x4_t(bw, sw2 + sw256(16 * s + 8 * (mi & 1) + x7, 8 * half + 2 * jp + (mi >> 1)));
          mma16816(accx[2 * jp], ad[s], bw[0], bw[1]);
          mma16816(accx[2 * jp + 1], ad[s], bw[2], bw[3]);
        }
      }
      if (half == 0) {                                             // the dh^T tile is dead: dh's rows overwrite it, then leave
        __syncwarp();
#pragma unroll
        for (int s = 0; s < 8; ++s)
#pragma unroll
          for (int i = 0; i < 4; ++i) sts32(tbh + sw256(g8 + 8 * (i & 1), 2 * s + (i >> 1)) + 4 * q4, ad[s][i]);
        __syncwarp();
        flush_tile<16>(tbh, p.dh + row0 * 128, 0, nv, lane);
        __syncwarp();
      }
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) {
        const int j = 8 * half + jj;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint32_t pr = preP[j][h];
          dpre[j][h] = pack_bf16(accx[jj][2 * h] * gelu_grad_f(bf16lo(pr)), accx[jj][2 * h + 1] * gelu_grad_f(bf16hi(pr)));
        }
      }
    }
    // dpre leaves through the same tile
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
      for (int h = 0; h < 2; ++h) sts32(tbh + sw256(g8 + 8 * h, j) + 4 * q4, dpre[j][h]);
    __syncwarp();
    flush_tile<16>(tbh, p.dpre + row0 * 128, 0, nv, lane);
    __syncwarp();

    // ---- dquery: the column sums of dpre over the 16 rows (thread-local over h, then a reduce-scatter over the eight row lanes; invalid rows hold exact zeros)
    {
      float v[32];
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        v[2 * j] = bf16lo(dpre[j][0]) + bf16lo(dpre[j][1]);
        v[2 * j + 1] = bf16hi(dpre[j][0]) + bf16hi(dpre[j][1]);
      }
      float w16[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) {                               // xor 16: lanes whose g8 bit 2 is set keep v[16 ..], the others v[0 .. 15]
        const float send = g16 ? v[i] : v[i + 16], keep = g16 ? v[i + 16] : v[i];
        w16[i] = keep + __shfl_xor_sync(0xffffffffu, send, 16);
      }
      float w8[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {                                // xor 8
        const float send = g8b ? w16[i] : w16[i + 8], keep = g8b ? w16[i + 8] : w16[i];
        w8[i] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {                                // xor 4
        const float send = g4 ? w8[i] : w8[i + 4], keep = g4 ? w8[i + 4] : w8[i];
        dqp[i] += keep + __shfl_xor_sync(0xffffffffu, send, 4);
      }
    }

    // ---- dedge = dpre W1e in two halves of 64 channels
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      float accy[8][4];
#pragma unroll
      for (int j = 0; j < 8; ++j) { accy[j][0] = accy[j][1] = accy[j][2] = accy[j][3] = 0.f; }
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const uint32_t adp[4] = {dpre[2 * s][0], dpre[2 * s][1], dpre[2 * s + 1][0], dpre[2 * s + 1][1]};
#pragma unroll
        for (int jp = 0; jp < 4; ++jp) {
          uint32_t bw[4];
          ldsm_x4_t(bw, sw1 + sw256(16 * s + 8 * (mi & 1) + x7, 8 * half + 2 * jp + (mi >> 1)));
          mma16816(accy[2 * jp], adp, bw[0], bw[1]);
          mma16816(accy[2 * jp + 1], adp, bw[2], bw[3]);
        }
      }
#pragma unroll
      for (int jj = 0; jj < 8; ++jj)
#pragma unroll
        for (int h = 0; h < 2; ++h) sts32(tbh + sw256(g8 + 8 * h, 8 * half + jj) + 4 * q4, pack_bf16(accy[jj][2 * h], accy[jj][2 * h + 1]));
      __syncwarp();
      flush_tile<8>(tbh, p.dedge + row0 * 128, 8 * half, nv, lane);
      __syncwarp();
    }

    if (++tg == ntl) {                                             // the group is complete: dquery[g] (bf16)
      tg = 0;
      // lane (g8, q4) holds the sums of indices idx = 8 g8b' ... : after the three steps it kept v-index set { 16 g16 + 8 g8b + 4 g4 + i } (i = 0..3), idx = 2 j + e <-> o = 8 j + 2 q4 + e
      const int base = 16 * (g16 ? 1 : 0) + 8 * (g8b ? 1 : 0) + 4 * (g4 ? 1 : 0);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int idx = base + i;
        p.dquery[g * 128 + 8 * (idx >> 1) + 2 * q4 + (idx & 1)] = __float2bfloat16_rn(dqp[i]);
        dqp[i] = 0.f;
      }
    }
  }

  // ---- the bias gradient: the warps of the CTA in warp order
  cp_async_wait<0>();
  __syncthreads();
  float* part = reinterpret_cast<float*>(smem_raw);                // the weights are dead: [NW][128] floats
  {
    const int base = (q4 >> 1) * 8 + (q4 & 1) * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int idx = base + i;
      part[warp * 128 + 16 * (idx >> 1) + g8 + 8 * (idx & 1)] = dbp[i];
    }
  }
  __syncthreads();
  for (int o = tid; o < 128; o += C::NTHR) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < C::NW; ++w) s += part[w * 128 + o];
    p.db2_part[(size_t)blockIdx.x * 128 + o] = s;
  }
}

// nb_grad[j] = sum over the edges e pointing at node j of dpre[e] (rows of 128 bf16), in the order of ``perm`` (the edges sorted stably by destination: ascending edge id, a fixed summation order);
// ``off`` [NN + 1] the segment starts.  One warp per node, 4 channels per lane, 8 rows in flight.  bf16 out (the rounding the Triton path applies after its fp32 accumulation).
__global__ void __launch_bounds__(256) nb_reduce_kernel(const bf* __restrict__ dpre, const int64_t* __restrict__ perm, const int64_t* __restrict__ off, bf* __restrict__ out, int64_t nn) {
  const int64_t node = (int64_t)blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (node >= nn) return;
  const int64_t b = off[node], e = off[node + 1];
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
  int64_t i = b;
  for (; i + 8 <= e; i += 8) {
    uint2 v[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) v[u] = __ldg(reinterpret_cast<const uint2*>(dpre + perm[i + u] * 128) + lane);
#pragma unroll
    for (int u = 0; u < 8; ++u) { a0 += bf16lo(v[u].x); a1 += bf16hi(v[u].x); a2 += bf16lo(v[u].y); a3 += bf16hi(v[u].y); }
  }
  for (; i < e; ++i) {
    const uint2 v = __ldg(reinterpret_cast<const uint2*>(dpre + perm[i] * 128) + lane);
    a0 += bf16lo(v.x); a1 += bf16hi(v.x); a2 += bf16lo(v.y); a3 += bf16hi(v.y);
  }
  uint2 r;
  r.x = pack_bf16(a0, a1);
  r.y = pack_bf16(a2, a3);
  reinterpret_cast<uint2*>(out + node * 128)[lane] = r;
}

}  // namespace mp80
