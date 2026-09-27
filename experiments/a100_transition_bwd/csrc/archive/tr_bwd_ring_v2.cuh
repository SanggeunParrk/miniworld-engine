// tr_bwd_ring_sm80.cuh -- the Transition backward as ONE launch with every product computed once (16 M D H): two roles that hand the
// SwiGLU-backward intermediates over through L2.
//
//   DXP CTAs (0 .. ndxp - 1): 256-row tiles (8 warps x 32 rows): LayerNorm, [a | b] and dh GEMMs, the SwiGLU backward, d_xn (f16
//       accumulation, the DX role of tr_bwd_dx_sm80.cuh), the LayerNorm backward -> dx, dgamma / dbeta.  Per 32-hidden chunk they also
//       write h, 2 dA, dB (bf16, fragment-native 16 x 16 blocks) into their own ring of RING_K chunk slots in global memory and raise the
//       slot's READY count (one per warp).  A slot is overwritten only once its CONSUMED count says the W CTA read the previous chunk.
//       DXP CTA i walks the chunks of its tiles rotated by 4 (i % 4), so the four hidden slices become ready at evenly spread times.
//   W CTAs (the rest: 4 slices x nrep replicas): the weight gradients of one 128-unit slice (tr_bwd_w_sm80.cuh's warp layout, 192 f32
//       accumulators).  Work item = (DXP CTA, tile round) in the order its slice becomes ready; replica r takes items r, r + nrep, ...
//       (static, so the partial sums are bit-reproducible).  An item is 4 stages of 64 rows: x (normalised in shared memory with the
//       statistics DXP wrote), dy and the slice's 4 chunks of h | 2 dA | dB, loaded with cp.async.cg (L2 only, never a stale L1 line).
// In flight: ndxp x RING_K x 48 KB (26 MB at 68 x 8) -- produced and consumed while still in L2.  Flags live in global memory and are
// zeroed before the launch; all CTAs must be resident (one per SM).
#pragma once
#include "tr_bwd_dx_sm80.cuh"
#include "tr_bwd_w_sm80.cuh"

#ifndef RING_K
#define RING_K 8
#endif

namespace a100 {

struct RingParams {
  const __nv_bfloat16* x;
  const __nv_bfloat16* dy;
  const __nv_bfloat16* wdx;    // DX weights [16 chunks][W1 | W3 | WX16]
  const float4* gb;            // LN affine float4 slots (forward layout)
  const float* gamma;
  const float* beta;
  float2* stats;               // [T] written by DXP, read by W
  __nv_bfloat16* dx;
  float* dgb;                  // [ndxp][256]
  float* part;                 // [nrep][3][512][128] (dWa part holds 2 dA sums)
  uint4* ring;                 // [ndxp][RING_K][3072] 16 B
  unsigned int* ready;         // [ndxp][RING_K]
  unsigned int* consumed;      // [ndxp][RING_K]
  int T, ndxp, nrep;
  float eps;
  unsigned long long* prof;     // RING_PROF: [grid][4] (wait cycles, total cycles, items, -)
};

constexpr int CHUNK_U4 = 3072;                                  // 256 rows x 32 hidden x 3 matrices x 2 B / 16 B
DEVI unsigned ld_acquire(const unsigned* p) {
  unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v;
}
DEVI void red_release(unsigned* p, unsigned v) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory");
}
DEVI void wait_geq(const unsigned* p, unsigned v) {
  if (ld_acquire(p) >= v) return;
  while (ld_acquire(p) < v) __nanosleep(64);
}
DEVI void stg64(void* p, uint32_t a, uint32_t b) { asm volatile("st.global.v2.u32 [%0], {%1,%2};\n" ::"l"(p), "r"(a), "r"(b) : "memory"); }

// ================================================================================================================== DXP
DEVI void dxp_tile(const RingParams& p, const Ring<CfgDX>& w, uint32_t s_u, uint32_t sGB, int cta, int j, int r0, int total, int rot) {
  using G = CfgDX;
  constexpr int D = 128;
  const int lane = w.lane, g8 = lane >> 2, q = lane & 3, warp = threadIdx.x >> 5;
  // ---- LayerNorm (the forward's arithmetic) into xn A fragments, dy A fragments; quads built whole (see p_tile)
  uint32_t fa[2][8][4], fd[2][8][4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    uint32_t xnw[2][16], dyw[2][16];
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
      const int r = r0 + 16 * mt + 8 * hr + g8;
      uint4 xin[4];
      const uint32_t* dyr = reinterpret_cast<const uint32_t*>(p.dy + (size_t)r * D + 8 * q);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        xin[i] = ldg_nc_na(p.x + (size_t)r * D + 32 * i + 8 * q);
#pragma unroll
        for (int k = 0; k < 4; ++k) dyw[hr][4 * i + k] = __ldg(dyr + 16 * i + k);
      }
      float xv[32];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t wv[4] = {xin[i].x, xin[i].y, xin[i].z, xin[i].w};
#pragma unroll
        for (int k = 0; k < 4; ++k) { xv[8 * i + 2 * k] = bf16lo(wv[k]); xv[8 * i + 2 * k + 1] = bf16hi(wv[k]); }
      }
      float sm = 0.f;
#pragma unroll
      for (int e = 0; e < 32; ++e) sm += xv[e];
      const float mean = quad_sum(sm) * (1.f / D);
      float sq = 0.f;
#pragma unroll
      for (int e = 0; e < 32; ++e) { xv[e] -= mean; sq = fmaf(xv[e], xv[e], sq); }
      const float rstd = rsqrtf(quad_sum(sq) * (1.f / D) + p.eps);
      if (q == 0) p.stats[r] = make_float2(mean, rstd);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const uint4 gi = lds128(sGB + (i * 4 + q) * 16), bi = lds128(sGB + 512 + (i * 4 + q) * 16);
        const float4 gv = *reinterpret_cast<const float4*>(&gi), bv = *reinterpret_cast<const float4*>(&bi);
        xnw[hr][2 * i] = pack_bf16(fmaf(xv[4 * i] * rstd, gv.x, bv.x), fmaf(xv[4 * i + 1] * rstd, gv.y, bv.y));
        xnw[hr][2 * i + 1] = pack_bf16(fmaf(xv[4 * i + 2] * rstd, gv.z, bv.z), fmaf(xv[4 * i + 3] * rstd, gv.w, bv.w));
      }
    }
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      fa[mt][s][0] = xnw[0][2 * s]; fa[mt][s][1] = xnw[1][2 * s]; fa[mt][s][2] = xnw[0][2 * s + 1]; fa[mt][s][3] = xnw[1][2 * s + 1];
      fd[mt][s][0] = dyw[0][2 * s]; fd[mt][s][1] = dyw[1][2 * s]; fd[mt][s][2] = dyw[0][2 * s + 1]; fd[mt][s][3] = dyw[1][2 * s + 1];
    }
  }
  uint32_t acc[2][16][2];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int jj = 0; jj < 16; ++jj) acc[mt][jj][0] = acc[mt][jj][1] = 0u;
  const int lgr = (lane >> 3) & 1;
  const uint32_t w1x2 = lgr * 1024 + (lane & 7) * 16, w3x2 = G::OFF_W3 + lgr * 512 + (lane & 7) * 16;
  const uint32_t bx_off = G::OFF_WX + (lane >> 4) * 512 + ((((lane >> 3) & 1) << 3) + (lane & 7)) * 16;
  // this warp's blocks in a chunk slot: row block Rl = 2 warp + mt, k-block ps, matrix m -> ((Rl * 2 + ps) * 3 + m) * 32 + lane
  uint4* const ring0 = p.ring + (size_t)cta * RING_K * CHUNK_U4 + (2 * warp) * 6 * 32 + lane;
  unsigned* const rdy = p.ready + cta * RING_K;
  const unsigned* const cns = p.consumed + cta * RING_K;
#pragma unroll 1
  for (int n = 0; n < 16; ++n) {
    const int u = 16 * j + n, slot = u % RING_K, gen = u / RING_K;
#ifdef RING_PROF
    const unsigned long long tw0 = clock64();
#endif
    // at a slice's first chunk: the W item that read the 4 slots' previous occupants released them in order, so its last is enough
    if ((n & 3) == 0 && u + 3 >= RING_K && lane == 0) wait_geq(cns + (u + 3) % RING_K, (unsigned)((u + 3) / RING_K));
#ifdef RING_PROF
    if (lane == 0 && warp == 0) atomicAdd(p.prof + 4 * cta, clock64() - tw0);
#endif
    __syncwarp();
    uint4* const ob = ring0 + (size_t)slot * CHUNK_U4;
    bool pending;
    const uint32_t wb = w.begin(u, total, pending);
    (void)rot;
#pragma unroll
    for (int ps = 0; ps < 2; ++ps) {
      uint32_t hA[2][4], hB[2][4];
#pragma unroll
      for (int n2 = 0; n2 < 2; ++n2) {
        float acc1[2][2][4], accd[2][4];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int e = 0; e < 4; ++e) { acc1[mt][0][e] = acc1[mt][1][e] = accd[mt][e] = 0.f; }
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          uint32_t ba[2], bb[2], bd[2];
          ldsm_x2(ba, wb + w1x2 + ps * 512 + n2 * 128 + s * 2048);
          ldsm_x2(bb, wb + w1x2 + ps * 512 + n2 * 128 + 256 + s * 2048);
          ldsm_x2(bd, wb + w3x2 + ps * 256 + n2 * 128 + s * 1024);
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
            mma16816(acc1[mt][0], fa[mt][s], ba[0], ba[1]);
            mma16816(acc1[mt][1], fa[mt][s], bb[0], bb[1]);
            mma16816(accd[mt], fd[mt][s], bd[0], bd[1]);
          }
        }
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          uint32_t wH[2], wA[2], wB[2];
#pragma unroll
          for (int hp = 0; hp < 2; ++hp) {
            float vA[2], vB[2], vH[2];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
              const int e = 2 * hp + k;
              const float ap = acc1[mt][0][e], bv = acc1[mt][1][e], d = accd[mt][e];
              const float t = tanh_approx(ap), sp = 1.f + t, silu = ap * sp;
              vH[k] = silu * bv;
              vB[k] = d * silu;
              vA[k] = d * bv * fmaf(-silu, t, silu + sp);         // 2 dA
            }
            hA[mt][2 * n2 + hp] = pack_f16(vA[0], vA[1]);
            hB[mt][2 * n2 + hp] = pack_f16(vB[0], vB[1]);
            wH[hp] = pack_bf16(vH[0], vH[1]); wA[hp] = pack_bf16(vA[0], vA[1]); wB[hp] = pack_bf16(vB[0], vB[1]);
          }
          // fragment-native block words 2 n2, 2 n2 + 1 of (row block 2 warp + mt, k-block ps): h | 2 dA | dB
          uint4* const o = ob + ((mt * 2 + ps) * 3) * 32;
          stg64(reinterpret_cast<uint32_t*>(o) + 2 * n2, wH[0], wH[1]);
          stg64(reinterpret_cast<uint32_t*>(o + 32) + 2 * n2, wA[0], wA[1]);
          stg64(reinterpret_cast<uint32_t*>(o + 64) + 2 * n2, wB[0], wB[1]);
        }
      }
#pragma unroll
      for (int abk = 0; abk < 2; ++abk)
#pragma unroll
        for (int jj = 0; jj < 8; ++jj) {
          uint32_t bw[4];
          ldsm_x4_t(bw, wb + bx_off + abk * 8192 + jj * 1024 + ps * 256);
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
            mma16816_h(acc[mt][2 * jj], abk ? hB[mt] : hA[mt], bw[0], bw[1]);
            mma16816_h(acc[mt][2 * jj + 1], abk ? hB[mt] : hA[mt], bw[2], bw[3]);
          }
        }
    }
    __syncwarp();
    // at a slice's last chunk: one fence + release for its four chunks (W waits for this slot only)
    if ((n & 3) == 3 && lane == 0) { __threadfence(); red_release(rdy + slot, 1u); }
    w.end(u, pending);
  }

  // ---- LayerNorm backward + residual (as dx_tile)
  const uint32_t sGam = s_u + G::OFF_GAM, wslot = s_u + G::OFF_DGB + warp * 1024;
  float dgs[32], dbs[32];
#pragma unroll
  for (int k = 0; k < 32; ++k) dgs[k] = dbs[k] = 0.f;
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
      const int r = r0 + 16 * mt + 8 * hr + g8;
      const float2 st = p.stats[r];
      uint32_t xw[16], dw[16];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint4 xi = ldg_nc_na(p.x + (size_t)r * D + 32 * i + 8 * q), di = ldg_nc_na(p.dy + (size_t)r * D + 32 * i + 8 * q);
        xw[4 * i] = xi.x; xw[4 * i + 1] = xi.y; xw[4 * i + 2] = xi.z; xw[4 * i + 3] = xi.w;
        dw[4 * i] = di.x; dw[4 * i + 1] = di.y; dw[4 * i + 2] = di.z; dw[4 * i + 3] = di.w;
      }
      float xh[32], gd[32], s1 = 0.f, s2 = 0.f;
#pragma unroll
      for (int J = 0; J < 16; ++J) {
        const uint32_t col = 32 * (J >> 2) + 8 * q + 2 * (J & 3);
        const uint2 gi = lds64(sGam + col * 4);
        const float2 dn2 = unpack_f16(acc[mt][J][hr]);
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float xv = e ? bf16hi(xw[J]) : bf16lo(xw[J]), dn = e ? dn2.y : dn2.x;
          xh[2 * J + e] = (xv - st.x) * st.y;
          gd[2 * J + e] = __uint_as_float(e ? gi.y : gi.x) * dn;
          dgs[2 * J + e] = fmaf(dn, xh[2 * J + e], dgs[2 * J + e]);
          dbs[2 * J + e] += dn;
          s1 = fmaf(gd[2 * J + e], xh[2 * J + e], s1);
          s2 += gd[2 * J + e];
        }
      }
      const float ca = quad_sum(s1) * (1.f / D), cb = quad_sum(s2) * (1.f / D);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        uint32_t o[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int J = 4 * i + k;
          o[k] = pack_bf16(fmaf(gd[2 * J] - fmaf(xh[2 * J], ca, cb), st.y, bf16lo(dw[J])),
                           fmaf(gd[2 * J + 1] - fmaf(xh[2 * J + 1], ca, cb), st.y, bf16hi(dw[J])));
        }
        stg128(p.dx + (size_t)r * D + 32 * i + 8 * q, make_uint4(o[0], o[1], o[2], o[3]));
      }
    }
#pragma unroll
  for (int k = 0; k < 32; ++k)
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) {
      dgs[k] += __shfl_xor_sync(0xffffffffu, dgs[k], o);
      dbs[k] += __shfl_xor_sync(0xffffffffu, dbs[k], o);
    }
  if (g8 == 0) {
#pragma unroll
    for (int J = 0; J < 16; ++J)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const uint32_t col = 32 * (J >> 2) + 8 * q + 2 * (J & 3) + e;
        sts32(wslot + col * 4, __float_as_uint(__uint_as_float(lds32(wslot + col * 4)) + dgs[2 * J + e]));
        sts32(wslot + 512 + col * 4, __float_as_uint(__uint_as_float(lds32(wslot + 512 + col * 4)) + dbs[2 * J + e]));
      }
  }
}

DEVI void dxp_role(const RingParams& p, uint8_t* smem, int cta) {
  using G = CfgDX;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t s_u = smem_u32(smem);
  const int ntile = p.T / G::BM, grid = p.ndxp;
  const int n_items = ntile > cta ? (ntile - cta + grid - 1) / grid : 0, total = n_items * 16;
  const int rot = 4 * (cta & 3);
  // LN table after the ring's dgb slots: reuse CfgDX's layout plus 1 KB (fits: CfgDX::SMEM + 1 KB < the fused budget)
  const uint32_t sGB = s_u + G::SMEM;
  Ring<G> w;
  w.sW = s_u; w.bars = s_u + G::OFF_BAR; w.src = p.wdx + tid * 8; w.tid = tid; w.lane = lane; w.rot = rot;
  if (tid == 0)
    for (int s = 0; s < G::NST; ++s) { mbar_init(w.full(s), G::NTHR); mbar_init(w.empty(s), G::NWARP); }
  for (int k = tid; k < 128; k += G::NTHR) reinterpret_cast<float*>(smem + G::OFF_GAM)[k] = p.gamma[k];
  for (int k = tid; k < G::NWARP * 256; k += G::NTHR) reinterpret_cast<float*>(smem + G::OFF_DGB)[k] = 0.f;
  for (int k = tid; k < 64; k += G::NTHR) reinterpret_cast<float4*>(smem + G::SMEM)[k] = p.gb[k];
  __syncthreads();
  if (n_items > 0) {
    for (int c = 0; c < G::AHEAD && c < total; ++c) w.issue((c + rot) % 16, c);
#pragma unroll 1
    for (int it = 0; it < n_items; ++it) dxp_tile(p, w, s_u, sGB, cta, it, (cta + it * grid) * G::BM + 32 * warp, total, rot);
    cp_async_wait<0>();
  }
  __syncthreads();
  for (int k = tid; k < 256; k += G::NTHR) {
    float v = 0.f;
#pragma unroll
    for (int wi = 0; wi < G::NWARP; ++wi) v += reinterpret_cast<float*>(smem + G::OFF_DGB)[wi * 256 + k];
    p.dgb[(size_t)cta * 256 + k] = v;
  }
}

// ================================================================================================================== W
// item k of slice s (the order slice s becomes ready): round j = k / ndxp; within the round the DXP CTAs whose rotation class makes
// slice s ready first: class cls = (k % ndxp) / (ndxp / 4) -> CTAs i with (s - i % 4) mod 4 == cls, i.e. i % 4 = (s - cls) mod 4
DEVI void w_item(int s, int k, int ndxp, int& i, int& j) {
  j = k / ndxp;
  const int rem = k - j * ndxp, per = ndxp >> 2, cls = rem / per, m = rem - cls * per;
  i = 4 * m + ((s - cls) & 3);
}

DEVI void w_role(const RingParams& p, uint8_t* smem, int sl, int rr, int nrep) {
  using G = CfgW;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const uint32_t s_u = smem_u32(smem);
  float* sGB = reinterpret_cast<float*>(smem + G::NSTAGE * G::STAGE);
  for (int k = tid; k < 128; k += G::NTHR) { sGB[k] = p.gamma[k]; sGB[128 + k] = p.beta[k]; }
  const int ntile = p.T / 256, rounds = (ntile + p.ndxp - 1) / p.ndxp;
  const int n_items_all = rounds * p.ndxp;
  const int n_mine = n_items_all > rr ? (n_items_all - rr + nrep - 1) / nrep : 0;   // items rr, rr + nrep, ... (some past the end)
  // stage g of this CTA = item g / 4, 64-row quarter g % 4
  auto item_of = [&](int g, int& i, int& j, int& t) { w_item(sl, rr + (g >> 2) * nrep, p.ndxp, i, j); t = i + j * p.ndxp; };
  // slot of chunk c of item (i, j): position n = (c - rot_i) mod 16 in that CTA's walk, global index 16 j + n
  auto slot_of = [&](int i, int j, int c, int& slot, int& gen) {
    const int n = (c - 4 * (i & 3)) & 15, u = 16 * j + n;
    slot = u % RING_K; gen = u / RING_K;
  };
  const int xr = tid >> 4, xg = tid & 15;
  auto load_stage = [&](int g) {
    int i, j, t;
    item_of(g, i, j, t);
    const int row0 = t * 256 + (g & 3) * G::RS;
    const uint32_t buf = s_u + (g % G::NSTAGE) * G::STAGE;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int r = 16 * (k & 3) + xr;
      cp_async16_full(buf + (k >> 2) * G::TILE + swz_w(r, xg), ((k >> 2) ? p.dy : p.x) + (size_t)(row0 + r) * 128 + xg * 8);
    }
    // A granules: slot k = block warp + 8 k = matrix k / 4 (h, dA, dB -> ring matrices 0, 1, 2), chunk cc = k % 4, k-block warp / 4,
    // row block within the tile 4 (g % 4) + warp % 4
    const int Rl = 4 * (g & 3) + (warp & 3), kl = warp >> 2;
#pragma unroll
    for (int cc = 0; cc < 4; ++cc) {
      int slot, gen;
      slot_of(i, j, 4 * sl + cc, slot, gen);
      const uint4* src = p.ring + ((size_t)i * RING_K + slot) * CHUNK_U4 + ((Rl * 2 + kl) * 3) * 32 + lane;
#pragma unroll
      for (int m = 0; m < 3; ++m) {
        const int kslot = 4 * m + cc;                          // W's matrix order is dA, dB, h; the ring's is h, 2 dA, dB
        const int rm = m == 2 ? 0 : m + 1;
        cp_async16_full(buf + 2 * G::TILE + (warp + 8 * kslot) * 512 + lane * 16, src + rm * 32);
      }
    }
    if (tid < G::RS / 2) cp_async16_full(buf + 2 * G::TILE + G::ABLK + tid * 16, p.stats + row0 + 2 * tid);
  };
  auto item_ready = [&](int g) {                               // every thread waits (acquire) for the item's four chunks
    int i, j, t;
    item_of(g, i, j, t);
#ifdef RING_PROF
    const unsigned long long tw0 = clock64();
#endif
    {                                                          // the slice's last chunk carries the flag
      int slot, gen;
      slot_of(i, j, 4 * sl + 3, slot, gen);
      wait_geq(p.ready + i * RING_K + slot, 8u * (gen + 1));
    }
#ifdef RING_PROF
    if (tid == 0) { atomicAdd(p.prof + 4 * blockIdx.x, clock64() - tw0); atomicAdd(p.prof + 4 * blockIdx.x + 2, 1ull); }
#endif
  };
  auto item_release = [&](int g) {
    int i, j, t;
    item_of(g, i, j, t);
    for (int cc = 0; cc < 4; ++cc) {
      int slot, gen;
      slot_of(i, j, 4 * sl + cc, slot, gen);
      red_release(p.consumed + i * RING_K + slot, 1u);
    }
  };
  auto valid = [&](int g) { int i, j, t; item_of(g, i, j, t); return t < ntile; };
  auto normalise = [&](int g) {
    const uint32_t buf = s_u + (g % G::NSTAGE) * G::STAGE;
    float gm[8], bt[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) { gm[e] = sGB[xg * 8 + e]; bt[e] = sGB[128 + xg * 8 + e]; }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int r = 16 * k + xr;
      const uint2 stw = lds64(buf + 2 * G::TILE + G::ABLK + r * 8);
      const float mean = __uint_as_float(stw.x), rstd = __uint_as_float(stw.y);
      const uint32_t a = buf + swz_w(r, xg);
      const uint4 v = lds128(a);
      const uint32_t wv[4] = {v.x, v.y, v.z, v.w};
      uint32_t o[4];
#pragma unroll
      for (int e = 0; e < 4; ++e)
        o[e] = pack_bf16(fmaf((bf16lo(wv[e]) - mean) * rstd, gm[2 * e], bt[2 * e]), fmaf((bf16hi(wv[e]) - mean) * rstd, gm[2 * e + 1], bt[2 * e + 1]));
      sts128(a, make_uint4(o[0], o[1], o[2], o[3]));
    }
  };

  float acc[3][16][4];
#pragma unroll
  for (int m = 0; m < 3; ++m)
#pragma unroll
    for (int jj = 0; jj < 16; ++jj)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[m][jj][e] = 0.f;
  const int g0 = 3 * warp;
  const int nst = 4 * n_mine;
  __syncthreads();
  if (nst > 0 && valid(0)) { item_ready(0); load_stage(0); }
  cp_async_commit();
#pragma unroll 1
  for (int g = 0; g < nst; ++g) {
    const bool v = valid(g);
    cp_async_wait<0>();
    __syncthreads();                                            // stage g landed; stage g - 1 retired
    if ((g & 3) == 0 && g > 0 && valid(g - 1) && tid == 0) item_release(g - 1);   // previous item's chunks all read into smem
    if (g + 1 < nst && valid(g + 1)) {
      if (((g + 1) & 3) == 0) item_ready(g + 1);
      load_stage(g + 1);
    }
    cp_async_commit();
    if (!v) continue;
    normalise(g);
    __syncthreads();
    const uint32_t buf = s_u + (g % G::NSTAGE) * G::STAGE;
    if (g0 + 2 < 16) w_stage<0>(acc, buf, g0, lane);
    else if (g0 >= 16) w_stage<2>(acc, buf, g0, lane);
    else w_stage<1>(acc, buf, g0, lane);
  }
  cp_async_wait<0>();
  __syncthreads();
  if (nst > 0 && valid(nst - 1) && tid == 0) item_release(nst - 1);
  // partial sums (as tr_bwd_w_kernel)
  const int g8 = lane >> 2, q = lane & 3;
#pragma unroll
  for (int m = 0; m < 3; ++m) {
    const int g = g0 + m;
    float* out = p.part + (((size_t)rr * 3 + (g >> 3)) * 512 + 128 * sl + 16 * (g & 7)) * 128;
#pragma unroll
    for (int jj = 0; jj < 16; ++jj)
#pragma unroll
      for (int hr = 0; hr < 2; ++hr)
        *reinterpret_cast<float2*>(out + (size_t)(8 * hr + g8) * 128 + 8 * jj + 2 * q) = make_float2(acc[m][jj][2 * hr], acc[m][jj][2 * hr + 1]);
  }
}

constexpr int RING_SMEM = (CfgW::SMEM > CfgDX::SMEM + 1024) ? CfgW::SMEM : CfgDX::SMEM + 1024;

__global__ void __launch_bounds__(256, 1) tr_bwd_ring_kernel(const RingParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int b = blockIdx.x;
#ifdef RING_PROF
  const unsigned long long t0 = clock64();
#endif
  if (b < p.ndxp) dxp_role(p, smem, b);
  else { const int k = b - p.ndxp; w_role(p, smem, k & 3, k >> 2, p.nrep); }
#ifdef RING_PROF
  if (threadIdx.x == 0) p.prof[4 * b + 1] = clock64() - t0;
#endif
}

}  // namespace a100
