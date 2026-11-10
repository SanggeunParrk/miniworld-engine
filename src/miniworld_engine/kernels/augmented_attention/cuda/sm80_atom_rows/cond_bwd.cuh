// cond_bwd.cuh -- the conditioning backward of the AF3 local atom DiT block on A100 / sm_80 (dmod in the layout [dsc1 | dbi1 | dsckv | dbikv | dsc2 | dbi2 | dos | dts] of cond_fwd's mod; the bf16 module's
// autograd rounding points):
//   variants v = 0, 1, 2 (the three AdaLNs):  dcn_v = rn(dsc_v Wsc_v + dbi_v Wbi_v)  (one fp32 accumulation, K = 256);  cn_v = rn(g_v xhat);  dx_v = rn(LN_bwd(rn(dcn_v g_v)));  d g_v += sum_rows dcn_v xhat
//   variant 3 (the gates):  dcg = rn(dos Wos + dts Wts)
//   dc = the bf16 sum of dx_0, dx_kv, dx_2, dcg;  cn_0, cn_kv, cn_2 are written for the dWmod GEMMs
// A CTA of NW warps takes tiles of 16 NW rows; the transposed weights W_v^T [128 channels][256 units] stream through a three-stage cp.async ring in chunks of 64 units (16 KB), shared by the warps; a
// warp's A fragments (dmod rows, 8 consecutive units per thread and 32) are requested a chunk ahead. d g is accumulated in shared memory (lane sums scattered over the quad's rows).
#pragma once
#include "rows_layout.cuh"

namespace lr80 {

struct CondBwdParams {
  const __nv_bfloat16* dmod;                         // [M][1024]
  const __nv_bfloat16* c;                            // [M][128]
  const __nv_bfloat16* wt;                           // [4][128][256]: W_v^T for the variants (cn_0, cn_kv, cn_2, c)
  const float* gamma;                                // [3][128]
  __nv_bfloat16* dc;                                 // [M][128]
  __nv_bfloat16* cn;                                 // [3][M][128]
  float* dgamma;                                     // [3][128], accumulated
  int M, cross;
  float eps;
  int prefetch;                                      // 1: the next tile's rows are requested into L2
};

// NSTAGE_: chunks in flight (weights and the warps' dmod rows are requested NSTAGE_ - 1 chunks ahead); a warp's dmod fragments of a chunk (16 rows x 64 units, 2 KB) stage through shared memory
template <int NW_, int MINB_, int NSTAGE_ = 3>
struct CondBwdCfg {
  static constexpr int NW = NW_, NTHR = NW_ * 32, MINB = MINB_, NSTAGE = NSTAGE_, W_BYTES = 128 * 128, A_BYTES = 16 * 128;
  static constexpr int O_A = NSTAGE * W_BYTES, O_G = O_A + NW * NSTAGE * A_BYTES, O_DG = O_G + 3 * 128 * 4, SMEM = O_DG + 3 * 128 * 4;
};

DEVI int cb_variant_chunk(int j, int cross) { return (!cross && j >= 4) ? j + 4 : j; }          // the weight chunk (variant 4 v + unit chunk) of the j-th chunk of a tile

template <class G>
struct CbLoadW {                                                   // the weight chunk loader's index math, once per thread: pieces i = tid + k NTHR of a chunk (row i >> 3, piece i & 7)
  static constexpr int KN = 128 * 8 / G::NTHR;
  int src[KN];
  uint32_t dst[KN];
  DEVI void init(int tid) {
#pragma unroll
    for (int k = 0; k < KN; ++k) {
      const int i = tid + k * G::NTHR, r = i >> 3, ch = i & 7;
      src[k] = qf_channel(r) * 256 + ch * 8;
      dst[k] = w128_off(r, ch);
    }
  }
  DEVI void load(const CondBwdParams& p, uint32_t stage, int wj) const {                         // variant wj / 4, units 64 (wj % 4) .. + 63
    const size_t base = (size_t)(wj >> 2) * 128 * 256 + 64 * (wj & 3);
#pragma unroll
    for (int k = 0; k < KN; ++k) cp_async16(stage + dst[k], p.wt + base + src[k]);
  }
};

template <class G>
DEVI void cb_load_a(const CondBwdParams& p, uint32_t stage, int wj, int t0, int lane) {            // the warp's 16 rows x 64 units of dmod (variant wj / 4, units 64 (wj % 4) ..), zero rows past M
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int i = lane + 32 * k, r = i >> 3, ch = i & 7;
    const bool ok = t0 + r < p.M;
    cp_async16(stage + w128_off(r, ch), ok ? p.dmod + (size_t)(t0 + r) * 1024 + 256 * (wj >> 2) + 64 * (wj & 3) + ch * 8 : p.dmod, ok ? 16 : 0);
  }
}

template <class G>
__global__ void __launch_bounds__(G::NTHR, G::MINB) cond_bwd_kernel(const CondBwdParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const uint32_t sb = smem_u32(smem_raw);
  const float* sg = reinterpret_cast<const float*>(smem_raw + G::O_G);
  float* sdg = reinterpret_cast<float*>(smem_raw + G::O_DG);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, q4 = lane & 3;

  for (int i = tid; i < 3 * 128; i += G::NTHR) { reinterpret_cast<float*>(smem_raw + G::O_G)[i] = p.gamma[i]; sdg[i] = 0.f; }

  const int ntile = (p.M + 16 * G::NW - 1) / (16 * G::NW);
  const int mine = (int)blockIdx.x < ntile ? (ntile - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x : 0;
  const int nch = p.cross ? 16 : 12;
  const int total = mine * nch;
  const uint32_t sa = sb + G::O_A + warp * G::NSTAGE * G::A_BYTES;     // this warp's ring of dmod fragments
  CbLoadW<G> ldw;
  ldw.init(tid);
  int icj = 0, ij = 0, itl = 0, ist = 0;                           // the request stream: chunk, its index in the tile, the tile, the ring stage (counters, no divisions)
  auto issue = [&]() {                                             // the next chunk of this CTA's stream: the shared weights and this warp's rows
    if (icj < total) {
      const int wj = cb_variant_chunk(ij, p.cross), tile = (int)blockIdx.x + itl * (int)gridDim.x;
      ldw.load(p, sb + ist * G::W_BYTES, wj);
      cb_load_a<G>(p, sa + ist * G::A_BYTES, wj, (tile * G::NW + warp) * 16, lane);
    }
    cp_async_commit();
    ++icj;
    if (++ij == nch) { ij = 0; ++itl; }
    if (++ist == G::NSTAGE) ist = 0;
  };
#pragma unroll
  for (int s = 0; s < G::NSTAGE - 1; ++s) issue();

  uint32_t wb[2];                                                  // B-fragment address of the unit block kb' (32 units = chunks 4 kb' .. + 3) in row g8
#pragma unroll
  for (int kbp = 0; kbp < 2; ++kbp) wb[kbp] = w128_off(g8, 4 * kbp + q4);

  float mean[2], rstd[2];
  uint32_t dcs[2][4][4];                                           // the running dc: bf16 pairs [hh][group][j]
  float acc[4][4][4];
  int rr[2];
  bool rok[2];
  int jn = 0, tl = 0, cs = 0;                                      // the consumed chunk: its index in the tile, the tile, the ring stage
  for (int ci = 0; ci < total; ++ci) {
    cp_async_wait<G::NSTAGE - 2>();
    __syncthreads();                                               // chunk ci (weights and rows) is in shared memory; the stages of chunk ci - 1 are free
    const int tile = (int)blockIdx.x + tl * (int)gridDim.x;
    const int wj = cb_variant_chunk(jn, p.cross), v = wj >> 2, kc = wj & 3;
    const int t0 = (tile * G::NW + warp) * 16;
    issue();
    if (jn == 0) {                                                 // a new tile: rows, the statistics of c, the first A fragments
      if (p.prefetch && tl + 1 < mine) {                           // the next tile's dmod and c rows into L2: its loads then hit
        const int nr = ((tile + (int)gridDim.x) * G::NW + warp) * 16;
        pf_rows(p.dmod, 2048, 0, 16, nr, 16, p.M, lane);
        pf_rows(p.c, 256, 0, 2, nr, 16, p.M, lane);
      }
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) { rr[hh] = t0 + g8 + 8 * hh; rok[hh] = rr[hh] < p.M; }
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        float xh[4][8];
#pragma unroll
        for (int gq = 0; gq < 4; ++gq) unpack8(rok[hh] ? ldg128(p.c + (size_t)rr[hh] * 128 + 32 * gq + 8 * q4) : make_uint4(0u, 0u, 0u, 0u), xh[gq]);
        float s = 0.f;
#pragma unroll
        for (int gq = 0; gq < 4; ++gq)
#pragma unroll
          for (int i = 0; i < 8; ++i) s += xh[gq][i];
        const float m = quad_sum(s) * (1.f / 128.f);
        float vs = 0.f;
#pragma unroll
        for (int gq = 0; gq < 4; ++gq)
#pragma unroll
          for (int i = 0; i < 8; ++i) { const float d = xh[gq][i] - m; vs = fmaf(d, d, vs); }
        mean[hh] = m;
        rstd[hh] = 1.f / sqrtf(quad_sum(vs) * (1.f / 128.f) + p.eps);
      }
    }
    uint4 au[2][2];                                                // this chunk's A fragments from the warp's ring
    const uint32_t sas = sa + cs * G::A_BYTES;
#pragma unroll
    for (int kbp = 0; kbp < 2; ++kbp)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) au[kbp][hh] = lds128_ro(sas + w128_off(g8 + 8 * hh, 4 * kbp + q4));
    if (kc == 0) {
#pragma unroll
      for (int gq = 0; gq < 4; ++gq)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[gq][j][0] = acc[gq][j][1] = acc[gq][j][2] = acc[gq][j][3] = 0.f;
    }
    const uint32_t wst = sb + cs * G::W_BYTES;
#pragma unroll
    for (int kbp = 0; kbp < 2; ++kbp) {
      uint32_t a[2][4];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) { a[0][hh] = au[kbp][hh].x; a[0][2 + hh] = au[kbp][hh].y; a[1][hh] = au[kbp][hh].z; a[1][2 + hh] = au[kbp][hh].w; }
#pragma unroll
      for (int nt = 0; nt < 16; ++nt) {
        const uint4 b = lds128_ro(wst + nt * 8 * 128 + wb[kbp]);
        mma16816(acc[nt >> 2][nt & 3], a[0], b.x, b.y);
        mma16816(acc[nt >> 2][nt & 3], a[1], b.z, b.w);
      }
    }

    if (kc == 3) {                                                 // variant v is complete
      uint32_t dcn[2][4][4];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int gq = 0; gq < 4; ++gq)
#pragma unroll
          for (int j = 0; j < 4; ++j) dcn[hh][gq][j] = pk2(acc[gq][j][2 * hh], acc[gq][j][2 * hh + 1]);
      if (v < 3) {
        float dgp[32];
#pragma unroll
        for (int i = 0; i < 32; ++i) dgp[i] = 0.f;
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const size_t row = (size_t)rr[hh];
          float xh[4][8];
#pragma unroll
          for (int gq = 0; gq < 4; ++gq) unpack8(rok[hh] ? ldg128(p.c + row * 128 + 32 * gq + 8 * q4) : make_uint4(0u, 0u, 0u, 0u), xh[gq]);
#pragma unroll
          for (int gq = 0; gq < 4; ++gq)
#pragma unroll
            for (int i = 0; i < 8; ++i) xh[gq][i] = (xh[gq][i] - mean[hh]) * rstd[hh];
          uint32_t dcng[4][4];
#pragma unroll
          for (int gq = 0; gq < 4; ++gq) {
            const float* gp = sg + v * 128 + 32 * gq + 8 * q4;
            const float4 g0 = *reinterpret_cast<const float4*>(gp), g1 = *reinterpret_cast<const float4*>(gp + 4);
            const float gv[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
            uint32_t cnw[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const float d0 = bf16lo(dcn[hh][gq][e]), d1 = bf16hi(dcn[hh][gq][e]);
              dcng[gq][e] = pk2(d0 * gv[2 * e], d1 * gv[2 * e + 1]);
              dgp[8 * gq + 2 * e] += d0 * xh[gq][2 * e];
              dgp[8 * gq + 2 * e + 1] += d1 * xh[gq][2 * e + 1];
              cnw[e] = pk2(xh[gq][2 * e] * gv[2 * e], xh[gq][2 * e + 1] * gv[2 * e + 1]);
            }
            if (rok[hh] && v != 1) stg128(p.cn + ((size_t)(v == 0 ? 0 : 2) * p.M + row) * 128 + 32 * gq + 8 * q4, make_uint4(cnw[0], cnw[1], cnw[2], cnw[3]));
            if (rok[hh] && v == 1) stg128(p.cn + ((size_t)1 * p.M + row) * 128 + 32 * gq + 8 * q4, make_uint4(cnw[0], cnw[1], cnw[2], cnw[3]));
          }
          uint32_t dxv[4][4];
          ln_backward(xh, rstd[hh], dcng, dxv);
#pragma unroll
          for (int gq = 0; gq < 4; ++gq)
#pragma unroll
            for (int e = 0; e < 4; ++e) dcs[hh][gq][e] = (v == 0) ? dxv[gq][e] : padd(dcs[hh][gq][e], dxv[gq][e]);
        }
        reduce_scatter_g8<32>(dgp, lane);
        const int base = g8_base(lane, 32);
#pragma unroll
        for (int m = 0; m < 4; ++m) {
          const int idx = base + m;
          atomicAdd(sdg + v * 128 + 32 * (idx >> 3) + 8 * q4 + (idx & 7), dgp[m]);
        }
      } else {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
#pragma unroll
          for (int gq = 0; gq < 4; ++gq)
#pragma unroll
            for (int e = 0; e < 4; ++e) dcs[hh][gq][e] = padd(dcs[hh][gq][e], dcn[hh][gq][e]);
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
#pragma unroll
          for (int gq = 0; gq < 4; ++gq)
            if (rok[hh]) stg128(p.dc + (size_t)rr[hh] * 128 + 32 * gq + 8 * q4, make_uint4(dcs[hh][gq][0], dcs[hh][gq][1], dcs[hh][gq][2], dcs[hh][gq][3]));
      }
    }
    if (++jn == nch) { jn = 0; ++tl; }
    if (++cs == G::NSTAGE) cs = 0;
  }
  __syncthreads();
  for (int i = tid; i < 3 * 128; i += G::NTHR) atomicAdd(p.dgamma + i, sdg[i]);
}

}  // namespace lr80
