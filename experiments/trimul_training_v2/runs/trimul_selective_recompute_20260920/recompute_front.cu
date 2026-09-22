// Selective recomputation: no a/b output, no gated/masked epilogue.
// SPDX-License-Identifier: Apache-2.0
// Derived from Anthropic uplifting-biomolecular-modeling f4f62fa6592ae4938d49b1757bea0cfeff9f468e,
// native/pkg/v5/csrc/tmn_kernels.cuh (k1_body). Original vendored source unchanged.
// Changes: consume existing normalized input (LNM=0), preserve original training
// preactivation saves/rounding, TMA-store interleaved gate/projection as well as a/b.
#include "tmn_kernels.cuh"
namespace tmn { namespace sm90 {
struct SavedFrontParams { K1Params base; CUtensorMap tm_gate,tm_proj; float* rstd; };
static_assert(sizeof(K1Params)==512,"upstream K1 ABI");
static_assert(sizeof(SavedFrontParams)==832,"saved front ABI");
using FrontBase=K1Cfg<MWK1_CZ,MWK1_CH,false,MWK1_BI,MWK1_BJ,MWK1_NSLOT,MWK1_SKCH,MWK1_SCHED>;
struct SavedFrontCfg : FrontBase {
  static constexpr int SMEM=FrontBase::SMEM+FrontBase::SMEM_STAGE;
  static_assert(SMEM*MINB<=SMEM_LIMIT,"training saves exceed shared memory");
};
template <class G, bool HAS_MASK, int LNM, bool SAVE, bool EMITX = false>
TMN_DEVI void saved_front_body(const SavedFrontParams& tp) {
  const auto& p = tp.base;
  constexpr int CZ = G::CZ, KS = G::KS, NSLOT = G::NSLOT, SPB = G::SPB, NBLK = G::NBLK, SKCH = G::SKCH, SLOT_BYTES = G::SLOT_BYTES, BI = G::BI, BJ = G::BJ;
  static_assert(!(G::ZF32 && LNM == 2), "the reference-order LayerNorm is defined on bf16 inputs");
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* sA = smem;
  uint8_t* sW = smem + G::SMEM_A;
  uint8_t* sStage = sW + G::SMEM_W;
  uint8_t* sGate = sStage;
  uint8_t* sProj = sGate + G::SMEM_STAGE;
  float* sGamma = reinterpret_cast<float*>(sProj + G::SMEM_STAGE);
  float* sBeta = sGamma + CZ;
  uint64_t* bars = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sGamma) + G::SMEM_GB);
  uint64_t* barA_full = bars + 0;
  uint64_t* barA_empty = bars + 1;
  uint64_t* barW_full = bars + 2;               // [NSLOT]
  uint64_t* barW_empty = bars + 2 + NSLOT;       // [NSLOT]
  uint64_t* barGo = bars + G::NBAR;              // one-shot: warpgroup 0 -> warpgroup 1 start offset
  static_assert((G::NBAR + 1) * 8 <= G::SMEM_BAR, "barrier area");
  // start offset in weight blocks, clamped to what the ring can hold: with a streamed ring warpgroup 0 alone can retire at most NSLOT/SPB - 1 blocks
  // before it needs a slot that only both warpgroups together release (a larger offset would deadlock by construction)
  constexpr int K1OFF = TMN_K1_START_OFFSET <= 0 ? 0 : (G::W_RESIDENT || TMN_K1_START_OFFSET <= G::NSLOT / G::SPB - 1) ? TMN_K1_START_OFFSET : (G::NSLOT / G::SPB - 1);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wg = __shfl_sync(0xffffffffu, tid >> 7, 0);   // warp-uniform by construction: the register reallocation below (setmaxnreg) and the role split
                                                            // are per warp, and the allocator budgets each side of the branch by its setmaxnreg value
  if (tid == 0 && dyn_smem_size() < (uint32_t)G::SMEM) __trap();
  const int n_iter = (p.num_tiles - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;     // tiles blockIdx.x + it * gridDim.x
  for (int i = tid; i < CZ; i += G::NTHR) { sGamma[i] = p.gamma[i]; sBeta[i] = p.beta[i]; }
  if (tid == 0) {
    mbar_init(barA_full, 1); mbar_init(barA_empty, 4 * G::NCWG);           // one arrival per consumer warp
    for (int s = 0; s < NSLOT; ++s) { mbar_init(barW_full + s, 1); mbar_init(barW_empty + s, 4 * G::NCWG); }
    if (K1OFF > 0) mbar_init(barGo, 1);
    fence_barrier_init();
    tma_prefetch_desc(&p.tm_z); tma_prefetch_desc(&p.tm_w);

    tma_prefetch_desc(&tp.tm_gate); tma_prefetch_desc(&tp.tm_proj);
  }
  __syncthreads();

  if (wg == 0) {
    // ------------------------------------------------------------------ producer warpgroup
    setmaxnreg_dec<G::PROD_REGS>();
    if (warp == 0 && lane == 0) {                       // z tiles: one load per tile, issued as soon as the consumers have pulled the previous tile into registers
      for (int it = 0; it < n_iter; ++it) {
        const int tile = (int)blockIdx.x + it * (int)gridDim.x;
        if (it > 0) mbar_wait(barA_empty, (it - 1) & 1);
        const int i0 = (tile / p.tiles_j) * BI, j0 = (tile % p.tiles_j) * BJ;
        mbar_arrive_expect_tx(barA_full, G::SMEM_A);
#pragma unroll
        for (int kc = 0; kc < G::NKCA; ++kc) tma_load_3d(sA + kc * G::CHUNK_BYTES, &p.tm_z, barA_full, kc * G::CHUNK_CH, j0, i0);
      }
    } else if (warp == 1 && lane == 0) {                // weight ring: NBLK * SPB slot loads per tile, identical sequence every tile (L2-resident after the first)
      const uint32_t per_tile = (uint32_t)(NBLK * SPB);
      const uint32_t n_w = G::W_RESIDENT ? (n_iter > 0 ? per_tile : 0u) : (uint32_t)n_iter * per_tile;
      for (uint32_t w_iter = 0; w_iter < n_w; ++w_iter) {
        const int s = w_iter % NSLOT; const uint32_t u = w_iter / NSLOT;
        if (u > 0) mbar_wait(barW_empty + s, (u - 1) & 1);
        mbar_arrive_expect_tx(barW_full + s, SLOT_BYTES);
        const int hb = w_iter % per_tile, b = hb / SPB, h = hb % SPB;
#pragma unroll
        for (int kk = 0; kk < SKCH; ++kk) tma_load_2d(sW + s * SLOT_BYTES + kk * 8192, &p.tm_w, barW_full + s, (h * SKCH + kk) * 64, 64 * b);
      }
    }
    __syncwarp();
    return;
  }

  // ------------------------------------------------------------------ consumer warpgroups
  setmaxnreg_inc<G::CONS_REGS>();
  const int cw = wg - 1;                 // 0 .. NCWG-1 : token rows [64 cw, 64 cw + 64) of the tile
  const int wiw = warp & 3;              // warp within warpgroup: rows 16*wiw.. of the m64 tile
  const uint32_t sA_u = smem_u32(sA), sW_u = smem_u32(sW);
  const uint32_t stage_u = smem_u32(sStage) + (uint32_t)(cw * 8192);   // this warpgroup's 2 x 4 KB staging buffers [32 ch][64 tok]
  const size_t plane = (size_t)p.Np * (size_t)p.Np;
  const int rho0 = 64 * cw + 16 * wiw;                  // first of this warp's 16 token rows (tile-relative)
  const int rowA = rho0 + (lane >> 2), rowB = rowA + 8;  // this thread's accumulator rows
  // stmatrix role: lane -> (matrix idx, row rr): channel ch = 16 hq + 8 (idx>>1) + rr; token granule tg = 2 wiw + (idx&1) (8 tokens each), stored at tg ^ (ch & 7)
  const int idx = lane >> 3, rr = lane & 7;
  const int chq = 8 * (idx >> 1) + rr;
  const uint32_t sts_off0 = (uint32_t)chq * 128 + (uint32_t)(((2 * wiw + (idx & 1)) ^ (chq & 7)) * 16);
  const uint32_t sts_off1 = sts_off0 + 16 * 128;
  // store role: warp wiw stores channels 8 wiw .. 8 wiw + 7 of the block; instruction e (0,1): lanes 8k..8k+7 -> channel 8 wiw + 4 e + k, granule lane%8
  const int st_ch0 = 8 * wiw + (lane >> 3), st_g = lane & 7;
  const uint32_t ld_off0 = (uint32_t)st_ch0 * 128 + (uint32_t)((st_g ^ (st_ch0 & 7)) * 16);
  const uint32_t ld_off1 = (uint32_t)(st_ch0 + 4) * 128 + (uint32_t)((st_g ^ ((st_ch0 + 4) & 7)) * 16);
  const int bar_id = 1 + cw;                            // named barrier of this warpgroup
  uint32_t w_iter = 0;
  // TMA plane store (BJ % 64 == 0: this warpgroup's 64 tokens are one plane row segment): the elected thread stores the whole [32 ch][64 tok] staging
  // buffer of a block with one bulk-tensor store; the buffer written two blocks later is the same one, so the elected thread drains the previous
  // store's smem READ before each block's warpgroup barrier (issued a block earlier: normally complete already).
  constexpr bool TMAST = (TMN_K1_BULK_STORE != 0) && (BJ % 64 == 0);
  const bool st_elect = (wiw == 0) && (lane == 0);

  if (K1OFF > 0 && cw == 1 && n_iter > 0) mbar_wait(barGo, 0);   // start offset: warpgroup 0's first K1OFF weight blocks retire before warpgroup 1 starts
  for (int t_local = 0; t_local < n_iter; ++t_local) {
    const int tile = (int)blockIdx.x + t_local * (int)gridDim.x;
    const int i0 = (tile / p.tiles_j) * BI, j0 = (tile % p.tiles_j) * BJ;
    // ---- this thread's two rows: validity + mask; this lane's two store runs: pointers + predicates
    const int iA = i0 + rowA / BJ, jA = j0 + rowA % BJ, iB = i0 + rowB / BJ, jB = j0 + rowB % BJ;
    const bool vA = (iA < p.N) && (jA < p.N), vB = (iB < p.N) && (jB < p.N);
    float mA = vA ? 1.f : 0.f, mB = vB ? 1.f : 0.f;
    const int rho_g = 64 * cw + 8 * st_g;                                  // first token of this lane's store granule
    const int is_ = i0 + rho_g / BJ, js_ = j0 + rho_g % BJ;
    const bool st_ok = (is_ < p.Np) && (js_ < p.Np);
    const int st_n = min(8, p.Np - js_);                                   // valid tokens of this lane's 8-token granule (ragged planes)
    __nv_bfloat16* gp0 = p.ab + (size_t)st_ch0 * plane + (size_t)is_ * p.Np + js_;   // + 32 b * plane per block; + 4 * plane for e = 1
    const size_t blk_stride = 32 * plane, e_stride = 4 * plane;
    const int iw = i0 + (64 * cw) / BJ, jw = j0 + (64 * cw) % BJ;              // this warpgroup's plane row / first token (TMA store coordinates)

    // ---- A fragments: from the swizzled z tile, LayerNorm applied (bf16: in registers; fp32: from smem, output bf16)
    mbar_wait(barA_full, t_local & 1);
    uint32_t fa[KS][4];
    LnStats st;
    if (G::ZF32) {
      st = ln_rows_f32<KS, G::CHUNK_BYTES>(fa, sA_u, rowA, rowB, sGamma, sBeta, lane, p.eps);
    } else {
      load_frag_bf16<KS, G::CHUNK_BYTES>(fa, sA_u, rho0, lane);
    }
    // WAR across proxies: the reads above go through the generic proxy, the producer's refill of sA is an async-proxy (TMA) write.  The mbarrier
    // release alone does not order the two; fence.proxy.async does.
    fence_proxy_async();
    __syncwarp();
    if (lane == 0) mbar_arrive(barA_empty);              // the producer may refill sA with the next tile now
    if (!G::ZF32) {
      constexpr bool LNSER = KS >= 24;                   // the affine pass serialised at 24+ k-steps (96 live fragment registers): schedule only, same values
      if (LNM == 1) st = ln_fragment<KS, LNSER>(fa, sGamma, sBeta, lane, p.eps);
      else if (LNM == 4) st = ln_fragment<KS, false, math::TX>(fa, sGamma, sBeta, lane, p.eps);
      else if (LNM == 2) st = ln_stock<0, LNSER>(fa, sGamma, sBeta, lane, p.eps);
      else st = LnStats{0.f, 1.f, 0.f, 1.f};
    }
    if (EMITX) {                                          // the normalised rows as bf16 for K3's gate operand: 4-byte stores, 4 lanes cover 32 contiguous bytes
      const int q = lane & 3;
      __nv_bfloat16* xa = p.xz + ((size_t)iA * (size_t)p.xs_i + (size_t)jA * (size_t)p.xs_j) + 2 * q;
      __nv_bfloat16* xb = p.xz + ((size_t)iB * (size_t)p.xs_i + (size_t)jB * (size_t)p.xs_j) + 2 * q;
#pragma unroll
      for (int ks = 0; ks < KS; ++ks) {
        if (vA) { stg32(xa + 16 * ks, fa[ks][0]); stg32(xa + 16 * ks + 8, fa[ks][2]); }
        if (vB) { stg32(xb + 16 * ks, fa[ks][1]); stg32(xb + 16 * ks + 8, fa[ks][3]); }
      }
    }
    if (SAVE) {                                           // dormant save-intermediates: LN_in statistics per pair row (row owner lane of each quad)
      if (p.stats != nullptr && (lane & 3) == 0) {
        if (vA) { p.stats[(size_t)iA * p.N + jA] = st.mA; tp.rstd[(size_t)iA * p.N + jA] = st.rA; }
        if (vB) { p.stats[(size_t)iB * p.N + jB] = st.mB; tp.rstd[(size_t)iB * p.N + jB] = st.rB; }
      }
    }

    // ---- NBLK weight blocks of 32 channels (n64 = gate | proj); MMAs of block b+1 are issued before the epilogue of block b
    float acc0[32], acc1[32];
    auto slot_of = [&](uint32_t wi) -> int { return G::W_RESIDENT ? (int)(wi % (uint32_t)(NBLK * SPB)) : (int)(wi % NSLOT); };
    auto phase_of = [&](uint32_t wi) -> uint32_t { return G::W_RESIDENT ? 0u : ((wi / NSLOT) & 1u); };
    auto issue_block = [&](float (&ac)[32], uint32_t wi) {
      uint32_t dlo[SPB], dhi[SPB];
#pragma unroll
      for (int j = 0; j < SPB; ++j) {
        const int s = slot_of(wi + j);
        mbar_wait(barW_full + s, phase_of(wi + j));
        const uint64_t d = smem_desc(sW_u + s * SLOT_BYTES, 16, 1024, 1);
        dlo[j] = (uint32_t)d; dhi[j] = (uint32_t)(d >> 32);
      }
#pragma unroll
      for (int i = 0; i < 32; ++i) ac[i] = 0.f;
      fence_regs(ac);
      wgmma_fence();
      mma_chain<SKCH>(ac, fa, dlo, dhi);
      wgmma_commit();
    };
    auto epilogue = [&](float (&ac)[32], int b, uint32_t wi, bool release) {   // release: free the block's ring slots here (schedule 0) or not (already freed)
      fence_regs(ac);
      __syncwarp();
      if (K1OFF > 0 && cw == 0 && t_local == 0 && b == K1OFF - 1 && wiw == 0 && lane == 0) mbar_arrive(barGo);   // warpgroup 1 may start
      if (release && !G::W_RESIDENT) {
        if (lane == 0) {
#pragma unroll
          for (int j = 0; j < SPB; ++j) mbar_arrive(barW_empty + slot_of(wi + j));
        }
      }
      // The preactivation layout is unchanged: [g0,p0,g1,p1,...] x M.
      // Stage gate/projection separately and TMA-store through stride-2-channel maps.
      uint32_t pg[4][2], pp[4][2];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        pg[q][0] = pack_bf16(ac[4*q], ac[4*q+1]);
        pg[q][1] = pack_bf16(ac[4*q+2], ac[4*q+3]);
        pp[q][0] = pack_bf16(ac[4*(q+4)], ac[4*(q+4)+1]);
        pp[q][1] = pack_bf16(ac[4*(q+4)+2], ac[4*(q+4)+3]);

      }
      const uint32_t gbuf=smem_u32(sGate)+(uint32_t)(cw*8192+(b&1)*4096);
      const uint32_t pbuf=smem_u32(sProj)+(uint32_t)(cw*8192+(b&1)*4096);
      stsm_x4_t(gbuf+sts_off0,pg[0][0],pg[0][1],pg[1][0],pg[1][1]);
      stsm_x4_t(gbuf+sts_off1,pg[2][0],pg[2][1],pg[3][0],pg[3][1]);
      stsm_x4_t(pbuf+sts_off0,pp[0][0],pp[0][1],pp[1][0],pp[1][1]);
      stsm_x4_t(pbuf+sts_off1,pp[2][0],pp[2][1],pp[3][0],pp[3][1]);
      static_assert(TMAST,"Rematerialization covers contiguous 64-token tiles");
      fence_proxy_async();
      if (st_elect) tma_store_wait_read<0>();
      named_bar_sync(bar_id,128);
      if (st_elect) {
        tma_store_3d(&tp.tm_gate,sGate+cw*8192+(b&1)*4096,jw,iw,32*b);
        tma_store_3d(&tp.tm_proj,sProj+cw*8192+(b&1)*4096,jw,iw,32*b);
        tma_store_commit();
      }
    };

    if constexpr (G::SCHED == 1) {
      // ---- schedule 1: one block at a time — per-slot waits under the earlier slots' MMAs, full retire, every slot released before the epilogue
      (void)acc1; (void)issue_block;
#pragma unroll 1
      for (int b = 0; b < NBLK; ++b, w_iter += SPB) {
        static_for<SPB>([&](auto Jc) {
          constexpr int J = decltype(Jc)::value;
          const int s = slot_of(w_iter + J);
          mbar_wait(barW_full + s, phase_of(w_iter + J));
          const uint64_t d = smem_desc(sW_u + s * SLOT_BYTES, 16, 1024, 1);
          if (J == 0) {
#pragma unroll
            for (int i = 0; i < 32; ++i) acc0[i] = 0.f;
            fence_regs(acc0);
          }
          wgmma_fence();
          mma_group<SKCH, KS, J>(acc0, fa, (uint32_t)d, (uint32_t)(d >> 32));
          wgmma_commit();                                // a group per slot: the wait for the next slot never sits inside an open (uncommitted) MMA group
        });
        wgmma_wait<0>();
        if (!G::W_RESIDENT) {                            // every slot of the block back to the producer before any epilogue work (measured: earlier than
#pragma unroll                                           // inside the epilogue after its register fence is worth ~6 % of K1 at c_z 384)
          for (int j = 0; j < SPB; ++j) { __syncwarp(); if (lane == 0) mbar_arrive(barW_empty + slot_of(w_iter + j)); }
        }
        epilogue(acc0, b, w_iter, false);
      }
    } else {
    issue_block(acc0, w_iter);
#pragma unroll 1
    for (int b = 0; b < NBLK; b += 2, w_iter += 2 * SPB) {
      issue_block(acc1, w_iter + SPB);
      wgmma_wait<1>();
      epilogue(acc0, b, w_iter, true);
      if (b + 2 < NBLK) { issue_block(acc0, w_iter + 2 * SPB); wgmma_wait<1>(); }
      else { wgmma_wait<0>(); }
      epilogue(acc1, b + 1, w_iter + SPB, true);
    }
    }
    // the A fragments are read ASYNCHRONOUSLY by every block's wgmma: keep their registers allocated until the last group of this tile has retired
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) fence_regs(fa[ks]);
  }
  if (TMAST && st_elect) tma_store_wait_all();           // every bulk store of this CTA complete before its shared memory is released
}



}}
extern "C" __global__ __launch_bounds__(tmn::sm90::SavedFrontCfg::NTHR,tmn::sm90::SavedFrontCfg::MINB)
void mw_recompute_front(__grid_constant__ const tmn::sm90::SavedFrontParams p) {
 tmn::sm90::saved_front_body<tmn::sm90::SavedFrontCfg,true,MW_FUSED,MW_FUSED,MW_FUSED>(p);
}
