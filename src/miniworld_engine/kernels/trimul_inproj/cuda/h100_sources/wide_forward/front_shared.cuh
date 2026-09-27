// SPDX-License-Identifier: Apache-2.0
// MiniWorld K1 shared-operand extension of the packaged kernel; original attribution retained in tmn_kernels.cuh.
#include "tmn_kernels.cuh"
namespace tmn { namespace sm90 {
TMN_DEVI void mw_shared_mma(float (&v)[32],uint64_t a,uint64_t b,int ac){
 asm volatile("{.reg .pred p;setp.ne.b32 p,%34,0;wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},%32,%33,p,1,1,0,0;}" : "+f"(v[0]),"+f"(v[1]),"+f"(v[2]),"+f"(v[3]),"+f"(v[4]),"+f"(v[5]),"+f"(v[6]),"+f"(v[7]),"+f"(v[8]),"+f"(v[9]),"+f"(v[10]),"+f"(v[11]),"+f"(v[12]),"+f"(v[13]),"+f"(v[14]),"+f"(v[15]),"+f"(v[16]),"+f"(v[17]),"+f"(v[18]),"+f"(v[19]),"+f"(v[20]),"+f"(v[21]),"+f"(v[22]),"+f"(v[23]),"+f"(v[24]),"+f"(v[25]),"+f"(v[26]),"+f"(v[27]),"+f"(v[28]),"+f"(v[29]),"+f"(v[30]),"+f"(v[31]) : "l"(a),"l"(b),"r"(ac));
}
template <class G, bool HAS_MASK, int LNM, bool SAVE, bool EMITX = false, int MT = 0>
TMN_DEVI void mw_k1_shared(const K1Params& p) {
  constexpr int CZ = G::CZ, KS = G::KS, NSLOT = G::NSLOT, SPB = G::SPB, NBLK = G::NBLK, SKCH = G::SKCH, SLOT_BYTES = G::SLOT_BYTES, BI = G::BI, BJ = G::BJ;
  static_assert(!(G::ZF32 && LNM == 2), "the reference-order LayerNorm is defined on bf16 inputs");
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* sA = smem;
  uint8_t* sW = smem + G::SMEM_A;
  uint8_t* sStage = sW + G::SMEM_W;
  float* sGamma = reinterpret_cast<float*>(sStage + G::SMEM_STAGE);
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
    if (TMN_K1_BULK_STORE) tma_prefetch_desc(&p.tm_ab);
  }
  __syncthreads();
  if (tid == 0) { TMN_TS(p.stats, 0, clock64()); TMN_TS(p.stats, 4, tmn_globaltimer()); TMN_TS(p.stats, 6, (unsigned long long)n_iter); }

  if (wg == 0) {
    // ------------------------------------------------------------------ producer warpgroup
    setmaxnreg_dec<G::PROD_REGS>();
    if (warp == 0 && lane == 0) {                       // z tiles: one load per tile, issued as soon as the consumers have pulled the previous tile into registers
      pdl_wait();                                       // TMN_PDL: z (and mask) may still be written by the previous grid; the weight ring is already in flight
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
  constexpr uint32_t SWM = (TMN_K1_STORE32 != 0 && BJ == 32) ? 3u : 7u;   // staging swizzle: 64B pattern for the [32][2][32] store box, else 128B
  const uint32_t sts_off0 = (uint32_t)chq * 128 + (uint32_t)(((2 * wiw + (idx & 1)) ^ (chq & SWM)) * 16);
  const uint32_t sts_off1 = sts_off0 + 16 * 128;
  // store role: warp wiw stores channels 8 wiw .. 8 wiw + 7 of the block; instruction e (0,1): lanes 8k..8k+7 -> channel 8 wiw + 4 e + k, granule lane%8
  const int st_ch0 = 8 * wiw + (lane >> 3), st_g = lane & 7;
  const uint32_t ld_off0 = (uint32_t)st_ch0 * 128 + (uint32_t)((st_g ^ (st_ch0 & SWM)) * 16);
  const uint32_t ld_off1 = (uint32_t)(st_ch0 + 4) * 128 + (uint32_t)((st_g ^ ((st_ch0 + 4) & SWM)) * 16);
  const int bar_id = 1 + cw;                            // named barrier of this warpgroup
  uint32_t w_iter = 0;
  // TMA plane store (BJ % 64 == 0: this warpgroup's 64 tokens are one plane row segment): the elected thread stores the whole [32 ch][64 tok] staging
  // buffer of a block with one bulk-tensor store; the buffer written two blocks later is the same one, so the elected thread drains the previous
  // store's smem READ before each block's warpgroup barrier (issued a block earlier: normally complete already).
  // TMN_K1_STORE32 (experiment): BJ == 32 tiles store through a [32 tok][2 pair rows][32 ch] box; the staging row (128 B) already holds the
  // warpgroup's 64 tokens as row i (first 32) then row i+1 (next 32), which is exactly that box's linear order under the same 128B swizzle.
  constexpr bool TMAST = (TMN_K1_BULK_STORE != 0) && (BJ % 64 == 0 || (TMN_K1_STORE32 != 0 && BJ == 32));
  const bool st_elect = (wiw == 0) && (lane == 0);

  pdl_wait();                                                     // TMN_PDL: consumers read the mask and write the planes
  if (K1OFF > 0 && cw == 1 && n_iter > 0) mbar_wait(barGo, 0);   // start offset: warpgroup 0's first K1OFF weight blocks retire before warpgroup 1 starts
  for (int t_local = 0; t_local < n_iter; ++t_local) {
    const int tile = (int)blockIdx.x + t_local * (int)gridDim.x;
    const int i0 = (tile / p.tiles_j) * BI, j0 = (tile % p.tiles_j) * BJ;
    if (t_local == n_iter - 1 && tid == 128) pdl_trigger();       // TMN_PDL: the dependent grid may take SMs as CTAs retire
    // ---- this thread's two rows: validity + mask; this lane's two store runs: pointers + predicates
    const int iA = i0 + rowA / BJ, jA = j0 + rowA % BJ, iB = i0 + rowB / BJ, jB = j0 + rowB % BJ;
    const bool vA = (iA < p.N) && (jA < p.N), vB = (iB < p.N) && (jB < p.N);
    float mA = vA ? 1.f : 0.f, mB = vB ? 1.f : 0.f;
    if (HAS_MASK) {
      const size_t oA = (size_t)iA * p.ms_i + (size_t)jA * p.ms_j, oB = (size_t)iB * p.ms_i + (size_t)jB * p.ms_j;
      if constexpr (MT == 1) {              // bf16 mask (TMN_MASK_TEMPLATE, name field m2): no per-call fp32 cast pass
        const __nv_bfloat16* mk = reinterpret_cast<const __nv_bfloat16*>(p.mask);
        if (vA) mA = __bfloat162float(__ldg(mk + oA));
        if (vB) mB = __bfloat162float(__ldg(mk + oB));
      } else if constexpr (MT == 2) {       // bool / uint8 mask (m3)
        const unsigned char* mk = reinterpret_cast<const unsigned char*>(p.mask);
        if (vA) mA = (float)__ldg(mk + oA);
        if (vB) mB = (float)__ldg(mk + oB);
      } else {                              // fp32 mask: the original contract
        if (vA) mA = __ldg(p.mask + oA);
        if (vB) mB = __ldg(p.mask + oB);
      }
    }
    const int rho_g = 64 * cw + 8 * st_g;                                  // first token of this lane's store granule
    const int is_ = i0 + rho_g / BJ, js_ = j0 + rho_g % BJ;
    const bool st_ok = (is_ < p.Np) && (js_ < p.Np);
    const int st_n = min(8, p.Np - js_);                                   // valid tokens of this lane's 8-token granule (ragged planes)
    __nv_bfloat16* gp0 = p.ab + (size_t)st_ch0 * plane + (size_t)is_ * p.Np + js_;   // + 32 b * plane per block; + 4 * plane for e = 1
    const size_t blk_stride = 32 * plane, e_stride = 4 * plane;
    const int iw = i0 + (64 * cw) / BJ, jw = j0 + (64 * cw) % BJ;              // this warpgroup's plane row / first token (TMA store coordinates)
    // TMN_K1_WARPSTORE geometry: lane = channel of the block; granule e = tokens 64 cw + 16 wiw + 8 e .. +7 (this warp's own stmatrix granules 2 wiw + e)
    const int rho_w0 = 64 * cw + 16 * wiw, rho_w1 = rho_w0 + 8;
    const int is_w0 = i0 + rho_w0 / BJ, js_w0 = j0 + rho_w0 % BJ, is_w1 = i0 + rho_w1 / BJ, js_w1 = j0 + rho_w1 % BJ;
    const bool ok_w0 = (is_w0 < p.Np) && (js_w0 < p.Np), ok_w1 = (is_w1 < p.Np) && (js_w1 < p.Np);
    const int n_w0 = min(8, p.Np - js_w0), n_w1 = min(8, p.Np - js_w1);
    __nv_bfloat16* gw0 = p.ab + (size_t)lane * plane + (size_t)is_w0 * p.Np + js_w0;   // + 32 b * plane per block
    __nv_bfloat16* gw1 = p.ab + (size_t)lane * plane + (size_t)is_w1 * p.Np + js_w1;
    const uint32_t wl_off0 = (uint32_t)lane * 128 + (uint32_t)(((2 * wiw) ^ (lane & SWM)) * 16), wl_off1 = (uint32_t)lane * 128 + (uint32_t)(((2 * wiw + 1) ^ (lane & SWM)) * 16);

    // ---- A fragments: from the swizzled z tile, LayerNorm applied (bf16: in registers; fp32: from smem, output bf16)
    mbar_wait(barA_full, t_local & 1);
    if (tid == 128) { if (t_local == 0) TMN_TS(p.stats, 1, clock64()); if (t_local == n_iter - 1) TMN_TS(p.stats, 2, clock64()); }
    {
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
    // Retain input shared tile until the last WGMMA consumer finishes. //              // the producer may refill sA with the next tile now
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
        if (vA) { p.stats[2 * ((size_t)iA * p.N + jA)] = st.mA; p.stats[2 * ((size_t)iA * p.N + jA) + 1] = st.rA; }
        if (vB) { p.stats[2 * ((size_t)iB * p.N + jB)] = st.mB; p.stats[2 * ((size_t)iB * p.N + jB) + 1] = st.rB; }
      }
    }

    #pragma unroll
    for(int ks=0;ks<KS;++ks){int mat=lane>>3,r8=lane&7;
      stsm_x4(sA_u+(ks/4)*G::CHUNK_BYTES+swz128(rho0+r8+((mat&1)?8:0),(16*(ks%4)+((mat&2)?8:0))*2),fa[ks][0],fa[ks][1],fa[ks][2],fa[ks][3]);
    }
    fence_proxy_async();named_bar_sync(bar_id,128);
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
        if (!(TMN_WSKIP != 0 && G::W_RESIDENT && t_local > 0)) mbar_wait(barW_full + s, phase_of(wi + j));
        const uint64_t d = smem_desc(sW_u + s * SLOT_BYTES, 16, 1024, 1);
        dlo[j] = (uint32_t)d; dhi[j] = (uint32_t)(d >> 32);
      }
#pragma unroll
      for (int i = 0; i < 32; ++i) ac[i] = 0.f;
      fence_regs(ac);
      wgmma_fence();
      static_for<SPB>([&](auto Jc){constexpr int J=decltype(Jc)::value;
        static_for<SKCH>([&](auto Kc){constexpr int K=decltype(Kc)::value;
          static_for<4>([&](auto Qc){constexpr int Q=decltype(Qc)::value;
            mw_shared_mma(ac,smem_desc(sA_u+(J*SKCH+K)*G::CHUNK_BYTES+cw*8192+Q*32,16,1024,1),smem_desc(sW_u+slot_of(wi+J)*SLOT_BYTES+K*8192+Q*32,16,1024,1),J>0||K>0||Q>0);
          });
        });
      });
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
      uint32_t pk[4][2];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        float vA0 = math::gate(ac[4 * q + 0], ac[4 * (q + 4) + 0], mA), vA1 = math::gate(ac[4 * q + 1], ac[4 * (q + 4) + 1], mA);   // sigmoid(g) p m, fp32
        float vB0 = math::gate(ac[4 * q + 2], ac[4 * (q + 4) + 2], mB), vB1 = math::gate(ac[4 * q + 3], ac[4 * (q + 4) + 3], mB);
        if (TMN_K1_NO_VSEL == 0) {   // experiment TMN_K1_NO_VSEL: invalid rows already carry mA/mB == 0 (zero-filled OOB z rows give finite products)
          if (!vA) { vA0 = 0.f; vA1 = 0.f; }
          if (!vB) { vB0 = 0.f; vB1 = 0.f; }
        }
        pk[q][0] = pack_bf16(vA0, vA1); pk[q][1] = pack_bf16(vB0, vB1);
      }
      const uint32_t sbuf = stage_u + (uint32_t)((b & 1) * 4096);            // double-buffered: one warpgroup barrier per block
      stsm_x4_t(sbuf + sts_off0, pk[0][0], pk[0][1], pk[1][0], pk[1][1]);      // channels 0..15 of the block
      stsm_x4_t(sbuf + sts_off1, pk[2][0], pk[2][1], pk[3][0], pk[3][1]);      // channels 16..31
      if (TMAST && p.vec) {
        fence_proxy_async();                                                 // the stmatrix writes -> visible to the async proxy that reads the buffer
        if (st_elect) tma_store_wait_read<0>();                              // the store issued from the other buffer's twin a block ago has read it
        named_bar_sync(bar_id, 128);
        if (st_elect) { tma_store_3d(&p.tm_ab, sStage + cw * 8192 + (b & 1) * 4096, jw, iw, 32 * b); tma_store_commit(); }
      } else if (TMN_K1_WARPSTORE != 0) {              // warp-local: this warp reads back only the granules its own stmatrix wrote -> no warpgroup barrier
        __syncwarp();
        const uint4 v0 = lds128(sbuf + wl_off0), v1 = lds128(sbuf + wl_off1);
        const size_t bo = (size_t)b * blk_stride;
        if (p.vec) { if (ok_w0) stg128(gw0 + bo, v0); if (ok_w1) stg128(gw1 + bo, v1); }
        else { if (ok_w0) stg_ragged(gw0 + bo, v0, n_w0); if (ok_w1) stg_ragged(gw1 + bo, v1, n_w1); }
        __syncwarp();                                    // the buffer written two blocks later is this one: every lane has read it
      } else {
        named_bar_sync(bar_id, 128);
        const uint4 v0 = lds128(sbuf + ld_off0), v1 = lds128(sbuf + ld_off1);
        const size_t bo = (size_t)b * blk_stride;
        if (st_ok) {
          if (p.vec) { stg128(gp0 + bo, v0); stg128(gp0 + bo + e_stride, v1); }
          else { stg_ragged(gp0 + bo, v0, st_n); stg_ragged(gp0 + bo + e_stride, v1, st_n); }   // unpadded planes with Np % 8 != 0: element stores, j < Np
        }
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
          if (!(TMN_WSKIP != 0 && G::W_RESIDENT && t_local > 0)) mbar_wait(barW_full + s, phase_of(w_iter + J));
          const uint64_t d = smem_desc(sW_u + s * SLOT_BYTES, 16, 1024, 1);
          if (J == 0) {
#pragma unroll
            for (int i = 0; i < 32; ++i) acc0[i] = 0.f;
            fence_regs(acc0);
          }
          wgmma_fence();
          static_for<SKCH>([&](auto Kc){constexpr int K=decltype(Kc)::value;
            static_for<4>([&](auto Qc){constexpr int Q=decltype(Qc)::value;
              mw_shared_mma(acc0,smem_desc(sA_u+(J*SKCH+K)*G::CHUNK_BYTES+cw*8192+Q*32,16,1024,1),smem_desc(sW_u+s*SLOT_BYTES+K*8192+Q*32,16,1024,1),J>0||K>0||Q>0);
            });
          });
          wgmma_commit();
#if MW_K1_STREAM
          wgmma_wait<0>();__syncwarp();if(lane==0)mbar_arrive(barW_empty+s);
#endif
        });
        wgmma_wait<0>();
        if (!G::W_RESIDENT && !MW_K1_STREAM) {            // non-streamed slots retire together
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
    fence_proxy_async();__syncwarp();if(lane==0)mbar_arrive(barA_empty);
  }
  if (TMAST && st_elect) tma_store_wait_all();           // every bulk store of this CTA complete before its shared memory is released
  if (tid == 128) { TMN_TS(p.stats, 3, clock64()); TMN_TS(p.stats, 5, tmn_globaltimer()); }
}



}}
