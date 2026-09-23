// SPDX-License-Identifier: Apache-2.0
// Derived from Anthropic v5 K1. Reuses its TMA weight ring and register operand,
// replacing the epilogue with masked input-projection and gate derivatives.
#include "tmn_kernels.cuh"
#ifndef MW_GP_STREAM
#define MW_GP_STREAM 0
#endif
namespace tmn { namespace sm90 {
TMN_DEVI void gp_shared_mma(float (&d)[32],uint64_t a,uint64_t b,int ac){
 asm volatile("{.reg .pred p;setp.ne.b32 p,%34,0;wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},%32,%33,p,1,1,0,0;}" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]) : "l"(a),"l"(b),"r"(ac));
}
struct GPParams {K1Params base; CUtensorMap dl, dr, out;};
template <class G, bool HAS_MASK, int LNM, bool SAVE, bool EMITX = false>
TMN_DEVI void gp_body(const GPParams& ext) {
  const K1Params& p=ext.base;
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
    for(int c=0;c<G::NCWG;++c)mbar_init(bars+G::NBAR+1+c,1);
    fence_barrier_init();
    tma_prefetch_desc(&p.tm_z); tma_prefetch_desc(&p.tm_w);
    if (TMN_K1_BULK_STORE) tma_prefetch_desc(&p.tm_ab);
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
  uint32_t w_iter = 0; int dphase=0;
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
    if (HAS_MASK) {
      if (vA) mA = __ldg(p.mask + (size_t)iA * p.ms_i + (size_t)jA * p.ms_j);
      if (vB) mB = __ldg(p.mask + (size_t)iB * p.ms_i + (size_t)jB * p.ms_j);
    }
    const int rho_g = 64 * cw + 8 * st_g;                                  // first token of this lane's store granule
    const int is_ = i0 + rho_g / BJ, js_ = j0 + rho_g % BJ;
    const bool st_ok = (is_ < p.Np) && (js_ < p.Np);
    const int st_n = min(8, p.Np - js_);                                   // valid tokens of this lane's 8-token granule (ragged planes)
    __nv_bfloat16* gp0 = p.ab + (size_t)st_ch0 * plane + (size_t)is_ * p.Np + js_;   // + 32 b * plane per block; + 4 * plane for e = 1
    const size_t blk_stride = 32 * plane, e_stride = 4 * plane;
    const int iw = i0 + (64 * cw) / BJ, jw = j0 + (64 * cw) % BJ;              // this warpgroup's plane row / first token (TMA store coordinates)

    // ---- A fragments: from the swizzled z tile, LayerNorm applied (bf16: in registers; fp32: from smem, output bf16)
    mbar_wait(barA_full, t_local & 1);
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
      static_for<SPB>([&](auto JJ){constexpr int J=decltype(JJ)::value;
       static_for<SKCH>([&](auto KK){constexpr int K=decltype(KK)::value;
        static_for<4>([&](auto QQ){constexpr int q=decltype(QQ)::value;
         gp_shared_mma(ac,smem_desc(sA_u+(J*SKCH+K)*G::CHUNK_BYTES+cw*8192+q*32,16,1024,1),smem_desc(sW_u+slot_of(wi+J)*SLOT_BYTES+K*8192+q*32,16,1024,1),J>0||K>0||q>0);
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
      const int side=(32*b)/(2*CZ), cb=(32*b)%(2*CZ);
      uint64_t* dbar=bars+G::NBAR+1+cw;
      if(st_elect)tma_store_wait_read<0>();
      named_bar_sync(bar_id,128);
      if(st_elect){mbar_arrive_expect_tx(dbar,4096);tma_load_3d(sStage+cw*8192,side?&ext.dr:&ext.dl,dbar,jw,iw,cb);}
      mbar_wait(dbar,dphase);dphase^=1;
      uint32_t dp[4][2],dg[4][2];
#pragma unroll
      for(int q=0;q<4;++q){
       float pv[4],gv[4];
#pragma unroll
       for(int j=0;j<4;++j){
        int row=16*wiw+(lane>>2)+(j>=2?8:0),c=8*q+2*(lane&3)+(j&1);
        float mask=j<2?mA:mB;
        float raw=__bfloat162float(*reinterpret_cast<__nv_bfloat16*>(sStage+cw*8192+swz128(c,row*2)));
        float d=__bfloat162float(__float2bfloat16_rn(raw*mask));
        float g=math::sigmoid(__bfloat162float(__float2bfloat16_rn(ac[4*q+j])));
        float proj=__bfloat162float(__float2bfloat16_rn(ac[4*(q+4)+j]));
        pv[j]=d*g;gv[j]=((d*proj)*g)*(1-g);
       }
       dp[q][0]=pack_bf16(pv[0],pv[1]);dp[q][1]=pack_bf16(pv[2],pv[3]);
       dg[q][0]=pack_bf16(gv[0],gv[1]);dg[q][1]=pack_bf16(gv[2],gv[3]);
      }
      named_bar_sync(bar_id,128);
      stsm_x4_t(stage_u+sts_off0,dp[0][0],dp[0][1],dp[1][0],dp[1][1]);
      stsm_x4_t(stage_u+sts_off1,dp[2][0],dp[2][1],dp[3][0],dp[3][1]);
      stsm_x4_t(stage_u+4096+sts_off0,dg[0][0],dg[0][1],dg[1][0],dg[1][1]);
      stsm_x4_t(stage_u+4096+sts_off1,dg[2][0],dg[2][1],dg[3][0],dg[3][1]);
      fence_proxy_async();named_bar_sync(bar_id,128);
      if(st_elect){tma_store_3d(&ext.out,sStage+cw*8192,jw,iw,side*4*CZ+cb);tma_store_3d(&ext.out,sStage+cw*8192+4096,jw,iw,side*4*CZ+2*CZ+cb);tma_store_commit();}

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
          static_for<SKCH>([&](auto KK){constexpr int K=decltype(KK)::value;
           static_for<4>([&](auto QQ){constexpr int q=decltype(QQ)::value;
            gp_shared_mma(acc0,smem_desc(sA_u+(J*SKCH+K)*G::CHUNK_BYTES+cw*8192+q*32,16,1024,1),smem_desc(sW_u+s*SLOT_BYTES+K*8192+q*32,16,1024,1),J>0||K>0||q>0);
           });
          });
          wgmma_commit();
#if MW_GP_STREAM
          wgmma_wait<0>();__syncwarp();if(lane==0)mbar_arrive(barW_empty+s);
#endif
          // a group per slot: the wait for the next slot never sits inside an open (uncommitted) MMA group
        });
        wgmma_wait<0>();
        if (!G::W_RESIDENT && !MW_GP_STREAM) {                            // every slot of the block back to the producer before any epilogue work (measured: earlier than
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
    fence_proxy_async();__syncwarp();if(lane==0)mbar_arrive(barA_empty);
  }
  if (st_elect) tma_store_wait_all();           // every bulk store of this CTA complete before its shared memory is released
}



}}
