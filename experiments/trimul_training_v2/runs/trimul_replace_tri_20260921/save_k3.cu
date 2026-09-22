// SPDX-License-Identifier: Apache-2.0
// Anthropic f4f62fa K3; Miniworld training epilogue, optional LN statistics and output projection/gate saves.
// Derived by derive.py. Apache-2.0 upstream attribution retained.
#include "tmn_kernels.cuh"
namespace tmn { namespace sm90 {
struct RecomputeParams { K3Params base; const __nv_bfloat16* dropscale; CUtensorMap tm_lnin,tm_lnout; __nv_bfloat16 *lnin,*lnout; float *mean_in,*rs_in,*mean_out,*rs_out; CUtensorMap tm_proj,tm_gate; __nv_bfloat16 *proj,*gate; };
template <int KS, bool SERIAL = false, int CLS = math::REF>
TMN_DEVI LnStats ln_save_xhat(uint32_t (&fa)[KS][4], const float* sGamma, const float* sBeta, int lane, float eps, void* output, int row, int M) {
  constexpr float invC = 1.f / (16 * KS);
  float meanA, meanB;
  if (CLS == math::TX) {
    float sA_ = 0.f, sB_ = 0.f;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      sA_ += bf16lo(fa[ks][0]) + bf16hi(fa[ks][0]) + bf16lo(fa[ks][2]) + bf16hi(fa[ks][2]);
      sB_ += bf16lo(fa[ks][1]) + bf16hi(fa[ks][1]) + bf16lo(fa[ks][3]) + bf16hi(fa[ks][3]);
    }
    meanA = quad_sum(sA_) * invC; meanB = quad_sum(sB_) * invC;
  } else {                                              // the statement: group sums of the 4 values a k-step holds per row, balanced tree over k-steps, lane butterfly
    float gA[KS], gB[KS];
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      gA[ks] = __fadd_rn(__fadd_rn(bf16lo(fa[ks][0]), bf16hi(fa[ks][0])), __fadd_rn(bf16lo(fa[ks][2]), bf16hi(fa[ks][2])));
      gB[ks] = __fadd_rn(__fadd_rn(bf16lo(fa[ks][1]), bf16hi(fa[ks][1])), __fadd_rn(bf16lo(fa[ks][3]), bf16hi(fa[ks][3])));
    }
    meanA = math::ln_mean(quad_sum(math::tree_sum(gA)), invC); meanB = math::ln_mean(quad_sum(math::tree_sum(gB)), invC);
  }
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) fence_regs(fa[ks]);   // re-derive the fp32 values in each pass (2 ALU ops) instead of keeping them live
  float rA, rB;
  if (CLS == math::TX) {
    float vA = 0.f, vB = 0.f;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      float d;
      d = bf16lo(fa[ks][0]) - meanA; vA += d * d; d = bf16hi(fa[ks][0]) - meanA; vA += d * d;
      d = bf16lo(fa[ks][2]) - meanA; vA += d * d; d = bf16hi(fa[ks][2]) - meanA; vA += d * d;
      d = bf16lo(fa[ks][1]) - meanB; vB += d * d; d = bf16hi(fa[ks][1]) - meanB; vB += d * d;
      d = bf16lo(fa[ks][3]) - meanB; vB += d * d; d = bf16hi(fa[ks][3]) - meanB; vB += d * d;
    }
    rA = rsqrtf(quad_sum(vA) * invC + eps); rB = rsqrtf(quad_sum(vB) * invC + eps);
  } else {
    float gA[KS], gB[KS];
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      gA[ks] = __fadd_rn(math::ln_sq_acc(math::ln_sq(bf16lo(fa[ks][0]), meanA), bf16hi(fa[ks][0]), meanA), math::ln_sq_acc(math::ln_sq(bf16lo(fa[ks][2]), meanA), bf16hi(fa[ks][2]), meanA));
      gB[ks] = __fadd_rn(math::ln_sq_acc(math::ln_sq(bf16lo(fa[ks][1]), meanB), bf16hi(fa[ks][1]), meanB), math::ln_sq_acc(math::ln_sq(bf16lo(fa[ks][3]), meanB), bf16hi(fa[ks][3]), meanB));
    }
    rA = math::ln_rstd(quad_sum(math::tree_sum(gA)), invC, eps); rB = math::ln_rstd(quad_sum(math::tree_sum(gB)), invC, eps);
  }
  const float mrA = meanA * rA, mrB = meanB * rB; (void)mrA; (void)mrB;
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) fence_regs(fa[ks]);
  uint32_t chain[2] = {0u, 0u};                        // two interleaved chains: step ks waits for step ks-2 (two steps' loads in flight)
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    // Save pre-affine normalization without changing the forward operand.
    #pragma unroll
    for(int j=0;j<4;++j){int rr=row+8*(j&1),cc=ks*16+2*(lane&3)+8*(j>>1);float mu=(j&1)?meanB:meanA,rs=(j&1)?rB:rA;
     float a=__fmul_rn(__fsub_rn(bf16lo(fa[ks][j]),mu),rs),b=__fmul_rn(__fsub_rn(bf16hi(fa[ks][j]),mu),rs);
#if XHAT_FP32
     reinterpret_cast<float*>(output)[(size_t)cc*M+rr]=a;reinterpret_cast<float*>(output)[(size_t)(cc+1)*M+rr]=b;
#else
     reinterpret_cast<__nv_bfloat16*>(output)[(size_t)cc*M+rr]=__float2bfloat16_rn(a);reinterpret_cast<__nv_bfloat16*>(output)[(size_t)(cc+1)*M+rr]=__float2bfloat16_rn(b);
#endif
    }
    const uint32_t k0 = (uint32_t)(16 * ks + 2 * (lane & 3)) + (SERIAL ? zero_dep(chain[ks & 1]) : 0u);
    const float2 g0 = lds64f(smem_u32(sGamma) + 4u * k0), b0 = lds64f(smem_u32(sBeta) + 4u * k0);
    const float2 g1 = lds64f(smem_u32(sGamma) + 4u * k0 + 32u), b1 = lds64f(smem_u32(sBeta) + 4u * k0 + 32u);
    if (CLS == math::TX) {                             // y = x * (r g) + (b - mean r g)
      fa[ks][0] = pack_bf16(fmaf(bf16lo(fa[ks][0]), rA * g0.x, fmaf(-mrA, g0.x, b0.x)), fmaf(bf16hi(fa[ks][0]), rA * g0.y, fmaf(-mrA, g0.y, b0.y)));
      fa[ks][1] = pack_bf16(fmaf(bf16lo(fa[ks][1]), rB * g0.x, fmaf(-mrB, g0.x, b0.x)), fmaf(bf16hi(fa[ks][1]), rB * g0.y, fmaf(-mrB, g0.y, b0.y)));
      fa[ks][2] = pack_bf16(fmaf(bf16lo(fa[ks][2]), rA * g1.x, fmaf(-mrA, g1.x, b1.x)), fmaf(bf16hi(fa[ks][2]), rA * g1.y, fmaf(-mrA, g1.y, b1.y)));
      fa[ks][3] = pack_bf16(fmaf(bf16lo(fa[ks][3]), rB * g1.x, fmaf(-mrB, g1.x, b1.x)), fmaf(bf16hi(fa[ks][3]), rB * g1.y, fmaf(-mrB, g1.y, b1.y)));
    } else {                                             // y = fma((x - mean) r, g, b): the statement
      fa[ks][0] = pack_bf16(math::ln_affine(bf16lo(fa[ks][0]), meanA, rA, g0.x, b0.x), math::ln_affine(bf16hi(fa[ks][0]), meanA, rA, g0.y, b0.y));
      fa[ks][1] = pack_bf16(math::ln_affine(bf16lo(fa[ks][1]), meanB, rB, g0.x, b0.x), math::ln_affine(bf16hi(fa[ks][1]), meanB, rB, g0.y, b0.y));
      fa[ks][2] = pack_bf16(math::ln_affine(bf16lo(fa[ks][2]), meanA, rA, g1.x, b1.x), math::ln_affine(bf16hi(fa[ks][2]), meanA, rA, g1.y, b1.y));
      fa[ks][3] = pack_bf16(math::ln_affine(bf16lo(fa[ks][3]), meanB, rB, g1.x, b1.x), math::ln_affine(bf16hi(fa[ks][3]), meanB, rB, g1.y, b1.y));
    }
    chain[ks & 1] = fa[ks][0] ^ fa[ks][3];
  }
  return LnStats{meanA, rA, meanB, rB};
}
using Cfg=K3Cfg<128,256,0,MW_BI,MW_BJ,MW_SLOT,MW_ACC>;

// Store the existing BF16 affine LayerNorm result; no additional LN math,
// statistics, projection, or gate saves. Each row is emitted once.
template<int KS>
TMN_DEVI void emit_ln(const uint32_t (&f)[KS][4],const CUtensorMap* map,
                     __nv_bfloat16* out,uint8_t* stage,int iw,int jw,int n){
 const int lane=threadIdx.x%32,w=(threadIdx.x/32)%4,mat=lane/8;
#if MW_STORE_METHOD == 0
 #pragma unroll
 for(int c64=0;c64<KS/4;++c64){
  if(lane==0)tma_store_wait_read<0>();__syncwarp();
  #pragma unroll
  for(int slab=0;slab<4;++slab){int k=4*c64+slab;
   stsm_x4(smem_u32(stage)+swz128(lane%8+8*(mat&1),(2*slab+(mat>>1))*16),f[k][0],f[k][1],f[k][2],f[k][3]);
  }
  fence_proxy_async();__syncwarp();
  if(lane==0){tma_store_3d(map,stage,64*c64,jw+16*w,iw);tma_store_commit();}
 }
 if(lane==0)tma_store_wait_read<0>();__syncwarp();
#else
 #pragma unroll
 for(int k=0;k<KS;++k){
  #pragma unroll
  for(int j=0;j<4;++j){int row=jw+16*w+lane/4+8*(j&1),col=k*16+2*(lane%4)+8*(j>>1);
   if(iw<n && row<n)stg32(out+((size_t)iw*n+row)*(KS*16)+col,f[k][j]);
  }
 }
#endif
}
template <class G, int LNM, bool UPD = false>
TMN_DEVI void infer_k3_body(const RecomputeParams& tp) {
  const K3Params& p=tp.base;
  constexpr int CZ = G::CZ, CH = G::CH, KSG = G::KSG, KSP = G::KSP, NKG = G::NKG, NKP = G::NKP, NB = G::NB, NBW = G::NBW, BN = G::BN, NSLOT = G::NSLOT;
  constexpr int NKCZ = G::NKCZ, BI = G::BI, BJ = G::BJ, OB = G::OB, SLOT_BYTES = G::SLOT_BYTES, CHB = G::CHUNK_BYTES, ESZ = G::ESZ;
  constexpr bool SPLITN = G::SPLITN, ZF32 = G::ZF32, OUT32 = G::OUT32, STG32 = G::STG32, PRENORM = G::PRENORM, RESG = G::RESG, NORES = G::NORES;
  constexpr int OSZ = OUT32 ? 4 : 2;                     // output / residual element size
  static_assert(!(ZF32 && (LNM == 2 || LNM == 3)), "the reference-order LayerNorm is defined on bf16 inputs");
  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* sX = smem;
  uint8_t* sZ = sX + G::SMEM_X;
  uint8_t* sW = sZ + G::SMEM_Z;
  uint8_t* sOut = sW + G::SMEM_W;
  uint8_t* sProj=sOut+G::SMEM_OUT; uint8_t* sGate=sProj+G::SMEM_OUT;
  float* sGin = reinterpret_cast<float*>(sOut + G::SMEM_OUT*(1+2*MW_SAVE_PG*(MW_PG_METHOD==0)));
  float* sBin = sGin + CZ; float* sGout = sBin + CZ; float* sBout = sGout + CH;
  uint64_t* bars = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sGin) + G::SMEM_GB);
  uint64_t* barZ_full = bars;                 // [NKCZ]
  uint64_t* barX_full = bars + NKCZ;
  uint64_t* barX_empty = barX_full + 1;       // X released (8 consumer warps) as soon as the fragments sit in registers
  uint64_t* barZ_empty = barX_empty + 1;      // [NKCZ]: z chunk c released by each consumer warp after its last read (fragment load, or the residual
  uint64_t* barW_full = barZ_empty + NKCZ;    //         read of the block pair living in that chunk) -> refilled with the next tile's chunk meanwhile
  uint64_t* barW_empty = barW_full + NSLOT;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wg = __shfl_sync(0xffffffffu, tid >> 7, 0);   // warp-uniform by construction: the register reallocation below (setmaxnreg) and the role split
                                                            // are per warp, and the allocator budgets each side of the branch by its setmaxnreg value
  if (tid == 0 && dyn_smem_size() < (uint32_t)G::SMEM) __trap();
  const int n_iter = (p.num_tiles - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
  if (MW_FUSED) for (int i = tid; i < CZ; i += NTHREADS) { sGin[i] = p.gamma_in[i]; sBin[i] = p.beta_in[i]; }
  for (int i = tid; i < CH; i += NTHREADS) { sGout[i] = p.gamma_out[i]; sBout[i] = p.beta_out[i]; }
  // register split: 24 / 240 for the bf16-tile kernels (bulk-store epilogue); the fp32-tile / cast / pre-normalised modes keep 40 / 232 (their live set is larger)
  constexpr bool R240 = (TMN_K3_REGS_24_240 != 0) && !G::ZF32 && !G::OUT32 && !G::RESG && !G::PRENORM;
  if (tid == 0) {
    for (int kc = 0; kc < NKCZ; ++kc) mbar_init(barZ_full + kc, 1);
    mbar_init(barX_full, 1); mbar_init(barX_empty, 8);
    for (int kc = 0; kc < NKCZ; ++kc) mbar_init(barZ_empty + kc, 8);
    for (int s = 0; s < NSLOT; ++s) { mbar_init(barW_full + s, 1); mbar_init(barW_empty + s, G::W_CONSUMERS); }
    fence_barrier_init();
    tma_prefetch_desc(&p.tm_z); tma_prefetch_desc(&p.tm_x); tma_prefetch_desc(&p.tm_wg); tma_prefetch_desc(&p.tm_wo);
    if ((TMN_K3_BULK_STORE != 0) && !G::ZF32 && !G::OUT32 && !G::RESG && !G::PRENORM) tma_prefetch_desc(&p.tm_out);
  }
  __syncthreads();

  if (wg == 0) {
    // ================================================================== producers: warp 0 operand stage, warp 1 weight ring
    setmaxnreg_dec<(R240 ? 24 : 40)>();
    if (warp == 0 && lane == 0) {
      for (int t_local = 0; t_local < n_iter; ++t_local) {
        const int tile = (int)blockIdx.x + t_local * (int)gridDim.x;
        const int i0 = (tile / p.tiles_j) * BI, j0 = (tile % p.tiles_j) * BJ;
        if (t_local > 0) mbar_wait(barX_empty, (t_local - 1) & 1);
        mbar_arrive_expect_tx(barX_full, G::SMEM_X);
#pragma unroll
        for (int h = 0; h < G::NSUB; ++h) {              // sub-tile h = tokens 64h..64h+63 of the tile: [C_H ch][64 tok], 64-channel chunks of 8 KB
          const int ih = i0 + (64 * h) / BJ, jh = j0 + (64 * h) % BJ;
#pragma unroll
          for (int kx = 0; kx < G::NKCX; ++kx) tma_load_3d(sX + h * (CH * 128) + kx * 8192, &p.tm_x, barX_full, jh, ih, kx * 64);
        }
#pragma unroll 1
        for (int kc = 0; kc < NKCZ; ++kc) {
          if (t_local > 0) mbar_wait(barZ_empty + kc, (t_local - 1) & 1);
          mbar_arrive_expect_tx(barZ_full + kc, CHB);
          tma_load_3d(sZ + kc * CHB, &p.tm_z, barZ_full + kc, kc * G::CHUNK_CH, j0, i0);
        }
      }
    } else if (warp == 1 && lane == 0) {
      const uint32_t per_tile = 2u * NB;
      const uint32_t n_w = G::W_RESIDENT ? (n_iter > 0 ? per_tile : 0u) : (uint32_t)n_iter * per_tile;
      for (uint32_t seq = 0; seq < n_w; ++seq) {         // per output block b: the projection rows (W_o[32 b .., :]) then the gate rows (W_og[32 b .., :])
        const int s = seq % NSLOT; const uint32_t u = seq / NSLOT;
        if (u > 0) mbar_wait(barW_empty + s, (u - 1) & 1);
        const int b = (int)((seq % per_tile) >> 1), which = (int)(seq & 1u);
        if (which == 0) {
          mbar_arrive_expect_tx(barW_full + s, G::SLOTP);
#pragma unroll
          for (int kc = 0; kc < NKP; ++kc) tma_load_2d(sW + s * SLOT_BYTES + kc * 4096, &p.tm_wo, barW_full + s, kc * 64, BN * b);
        } else {
          mbar_arrive_expect_tx(barW_full + s, G::SLOTG);
#pragma unroll
          for (int kc = 0; kc < NKG; ++kc) tma_load_2d(sW + s * SLOT_BYTES + kc * 4096, &p.tm_wg, barW_full + s, kc * 64, BN * b);
        }
      }
    }
    __syncwarp();
    return;
  }

  // ================================================================== consumers
  setmaxnreg_inc<(R240 ? 240 : 232)>();
#ifdef TMN_DEV_PROF
  unsigned long long pf[12] = {0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull};
  long long pt0 = clock64();
#define PF(i) do { long long t1_ = clock64(); pf[i] += (unsigned long long)(t1_ - pt0); pt0 = t1_; } while (0)
#else
#define PF(i) do { } while (0)
#endif
  const int cw = wg - 1, wiw = warp & 3, mat = lane >> 3, r8 = lane & 7;
  const int tok0 = SPLITN ? 0 : 64 * cw;                 // this WG's first token (tile-relative)
  const uint32_t sZ_u = smem_u32(sZ), sX_u = smem_u32(sX) + (uint32_t)((tok0 / 64) * (CH * 128)), sW_u = smem_u32(sW);
  const uint32_t stg_u = smem_u32(sOut) + (uint32_t)((4 * cw + wiw) * OB);   // this warp's staging slice [16 tok][32 ch]
  const int rho0 = tok0 + 16 * wiw;                      // tile-relative first row of this warp
  const int rowA = rho0 + (lane >> 2), rowB = rowA + 8;   // this thread's accumulator rows (tile-relative)
  const int gq = lane >> 2, q2 = 2 * (lane & 3);         // accumulator row within the warp's 16 (rows gq, gq + 8), column pair base within an 8-column group
  // Staging slice per warp = [16 tok][64 ch] (a pair of output blocks) in z's dtype.  bf16: 128-B rows, granule ^= row & 7 (swz128); stmatrix x4 matrix
  // mi = lane/8 covers rows 8 (mi&1) + r8 and the 8 channels 32 h + 16 kb + 8 (mi>>1) (h = block parity in the pair); register q of the x4 holds the
  // C-fragment pair (row gq + 8 (q&1), cols 16 kb + 8 (q>>1) + q2).  fp32: 256-B rows, granule ^= row & 7.  The vector pass moves 16-byte granules
  // g = lane + 32 it: bf16 row g/8, channels 8 (g%8) ..; fp32 row g/16, channels 4 (g%16) ..  -> 128 / 256 contiguous bytes per token row.
  const int lrow = r8 + 8 * (mat & 1);
  constexpr int NGR = STG32 ? 8 : 4;                     // 16-byte granules per lane per block pair in the vector pass
  // TMA output store (bf16 z, bf16 out): the residual is added in registers (the pair's z fragments re-read with ldmatrix in the accumulator layout, which
  // for a 16-column slab is the A-fragment layout) before staging, and each warp's staged [16 tok][64 ch] slice leaves with one bulk-tensor store;
  // the slice is reused by the next pair once that store has READ it (waited for at the next pair's start, behind two blocks of MMAs).
  constexpr bool K3ST = (TMN_K3_BULK_STORE != 0) && !ZF32 && !OUT32 && !RESG && !PRENORM;
  uint32_t zdep = 0;
  for (int t_local = 0; t_local < n_iter; ++t_local) {
    const int tile = (int)blockIdx.x + t_local * (int)gridDim.x;
    const int i0 = (tile / p.tiles_j) * BI, j0 = (tile % p.tiles_j) * BJ;
    const int iw = i0 + tok0 / BJ, jw = j0 + tok0 % BJ;               // this WG's 64 tokens: row iw, columns jw .. jw+63

    // ---- projection operand: X sub-tile [C_H ch rows][64 tok] -> ldmatrix.trans -> release the X stage -> LN_out (before z is touched:
    //      one raw fragment set live at a time keeps the normalisation inside the register budget)
    uint32_t fx[KSP][4];
    PF(0);                                               // 0: loop head / previous tile tail
    mbar_wait(barX_full, t_local & 1);
    PF(1);                                               // 1: X wait
    {
      const int tokc = 16 * wiw + ((mat & 1) ? 8 : 0);
#pragma unroll
      for (int ks = 0; ks < KSP; ++ks) {
        const int krow = 16 * ks + r8 + ((mat & 2) ? 8 : 0);
        ldsm_x4_t(fx[ks], sX_u + swz128(krow, tokc * 2));
      }
      uint32_t dep = 0;                                  // release once every read has returned (one destination register per instruction feeds the
#pragma unroll                                           // dependency); the proxy fence orders these generic-proxy reads before the async-proxy refill
      for (int ks = 0; ks < KSP; ++ks) dep ^= fx[ks][0];
      dep = zero_dep(__reduce_or_sync(0xffffffffu, dep));
      fence_proxy_async();
      if (lane == 0) mbar_arrive_dep(barX_empty, dep);
    }
    PF(2);                                               // 2: X ldmatrix + release
    LnStats stats_out=ln_save_xhat<KSP,MW_SERIAL>(fx,sGout,sBout,lane,p.eps,tp.lnout,iw*p.N+jw+16*wiw+(lane>>2),p.N*p.N);
#if MW_SAVE_STATS_OUT
    if((!SPLITN||cw==0)&&(lane&3)==0 && iw<p.N){int r=jw+16*wiw+(lane>>2);if(r<p.N){tp.rs_out[(size_t)iw*p.N+r]=stats_out.rA;}if(r+8<p.N){tp.rs_out[(size_t)iw*p.N+r+8]=stats_out.rB;}}
#endif
#if MW_SAVE_OUT
    if(!SPLITN || cw==0)emit_ln(fx,&tp.tm_lnout,tp.lnout,sOut+(4*cw+wiw)*OB,iw,jw,p.N);
#endif
    PF(3);                                               // 3: LN_out
    // ---- gate operand: z rows -> release the z stage -> LN_in
    uint32_t fz[KSG][4];
    for (int kc = 0; kc < NKCZ; ++kc) mbar_wait(barZ_full + kc, t_local & 1);
    PF(4);                                               // 4: z wait
    if (ZF32) ln_rows_f32<KSG, CHB>(fz, sZ_u, rowA, rowB, sGin, sBin, lane, p.eps);   // fp32 statistics + normalisation from smem -> bf16 fragments
    else load_frag_bf16<KSG, CHB>(fz, sZ_u, rho0, lane);
    {   // release the z chunks this warp will not touch again (all of them without residual; else those whose block pairs another warpgroup finishes)
      uint32_t dep = 0;
#pragma unroll
      for (int ks = 0; ks < KSG; ++ks) dep ^= fz[ks][0] ^ fz[ks][3];
      dep = zero_dep(__reduce_or_sync(0xffffffffu, dep));
      fence_proxy_async();
      if (lane == 0) {
#pragma unroll
        for (int kc = 0; kc < NKCZ; ++kc) {
          const int pair_b0 = (kc * G::CHUNK_CH) / (2 * BN) * 2;                        // first block of the pair whose channels live in chunk kc
          const bool mine = SPLITN ? (((pair_b0 >> 1) & 1) == cw) : true;             // split-N: pairs alternate between the warpgroups
          const bool now = !(p.residual && mine && !RESG && !NORES);                                // residual word 0: every chunk at once
          if (now) mbar_arrive_dep(barZ_empty + kc, dep);   // RESG: the residual is the fp32 z in global, the tile is free now
        }
      }
    }
    PF(5);                                               // 5: z load + release
    LnStats stats_in=ln_fragment<KSG,MW_SERIAL>(fz,sGin,sBin,lane,p.eps);
#if MW_SAVE_STATS_IN
    if((!SPLITN||cw==0)&&(lane&3)==0 && iw<p.N){int r=jw+16*wiw+(lane>>2);if(r<p.N){tp.mean_in[(size_t)iw*p.N+r]=stats_in.mA;tp.rs_in[(size_t)iw*p.N+r]=stats_in.rA;}if(r+8<p.N){tp.mean_in[(size_t)iw*p.N+r+8]=stats_in.mB;tp.rs_in[(size_t)iw*p.N+r+8]=stats_in.rB;}}
#endif
#if MW_SAVE_IN
    if(!SPLITN || cw==0)emit_ln(fz,&tp.tm_lnin,tp.lnin,sOut+(4*cw+wiw)*OB,iw,jw,p.N);
#endif
    PF(6);                                               // 6: LN_in

    // ---- output blocks of 32 channels: block q+1's MMAs are in flight while block q's epilogue runs (two accumulator sets)
    float accP0[16], accG0[16], accP1[16], accG1[16];
    auto blk_of = [&](int qb) -> int { return SPLITN ? 4 * (qb >> 1) + 2 * cw + (qb & 1) : qb; };   // qb-th block of this warpgroup (split-N: pairs alternate)
    auto slot_of = [&](uint32_t seq) -> int { return G::W_RESIDENT ? (int)(seq % (2u * NB)) : (int)(seq % NSLOT); };
    auto phase_of = [&](uint32_t seq) -> uint32_t { return G::W_RESIDENT ? 0u : ((seq / NSLOT) & 1u); };
    auto issue = [&](float (&accP)[16], float (&accG)[16], int qb) {
      const uint32_t seqP = ((uint32_t)t_local * NB + (uint32_t)blk_of(qb)) * 2u, seqG = seqP + 1u;
      {
        const int s = slot_of(seqP); mbar_wait(barW_full + s, phase_of(seqP));
        PF(7);                                           // 7: W(P) wait
        const uint64_t d = smem_desc(sW_u + s * SLOT_BYTES, 16, 1024, 1);
#pragma unroll
        for (int i = 0; i < 16; ++i) accP[i] = 0.f;
        fence_regs(accP);
        wgmma_fence();
        mma_chain32<KSP>(accP, fx, (uint32_t)d, (uint32_t)(d >> 32));
        wgmma_commit();
      }
      {
        PF(8);                                           // 8: P issue
        const int s = slot_of(seqG); mbar_wait(barW_full + s, phase_of(seqG));
        PF(7);                                           // 7: W(G) wait
        const uint64_t d = smem_desc(sW_u + s * SLOT_BYTES, 16, 1024, 1);
#pragma unroll
        for (int i = 0; i < 16; ++i) accG[i] = 0.f;
        fence_regs(accG);
        wgmma_fence();
        mma_chain32<KSG>(accG, fz, (uint32_t)d, (uint32_t)(d >> 32));
        wgmma_commit();
        PF(8);                                           // 8: G issue
      }
    };
    // vector-pass geometry of this lane for the tile: granule `it` covers token row jl + RSTEP it, 16 bytes at channel byte 16 cgl of a block pair
    // (a staging granule = 16 B of the staged slice: 8 bf16 update values (modes 0, 2) or 4 fp32 sums (mode 1); in mode 2 it expands to 32 output bytes)
    constexpr int RSTEP = STG32 ? 2 : 4;                 // token rows between a lane's consecutive granules (16 | 8 lanes per row)
    constexpr int OGB = RESG ? 32 : 16;                  // output bytes per staging granule
    const int rl = 16 * wiw + (STG32 ? (lane >> 4) : (lane >> 3)), cgl = STG32 ? (lane & 15) : (lane & 7);   // WG-relative row, granule
    const int jl = jw + rl;
    const size_t orow_off = (((size_t)iw * p.N + jl) * CZ) * OSZ + (size_t)OGB * cgl;
    uint8_t* orow = reinterpret_cast<uint8_t*>(p.out) + orow_off;
    const uint8_t* zrow = reinterpret_cast<const uint8_t*>(p.zres) + orow_off;          // mode 2: the fp32 z rows (same geometry as the fp32 output)
    const int nvalid = iw < p.N ? (p.N - jl + RSTEP - 1) / RSTEP : 0;                     // granules it < nvalid are inside the ragged edge
    constexpr size_t GSTRIDE = (size_t)RSTEP * CZ * OSZ;                                 // output bytes between a lane's consecutive granule rows
    auto release = [&](int qb) {                         // this warp's MMAs of block qb have retired (wgmma wait + syncwarp): free its two weight slots
      __syncwarp();
      if (!G::W_RESIDENT && lane == 0) {
        const uint32_t seqP = ((uint32_t)t_local * NB + (uint32_t)blk_of(qb)) * 2u;
        mbar_arrive(barW_empty + slot_of(seqP)); mbar_arrive(barW_empty + slot_of(seqP + 1u));
      }
    };
    // o = bf16(sigmoid(g) * p) in the accumulator layout -> this warp's staging slice, half h of the pair
    // Same BF16 projection/logit and FP32 epilogue as saved training K3.
    // Original input z stays in shared memory until the residual is consumed.
    auto stage = [&](float (&accP)[16], float (&accG)[16], int h, int b0) {
      static_assert(K3ST && MW_FUSED && !UPD, "BF16 fused training epilogue only");
      fence_regs(accP); fence_regs(accG);
      uint32_t fr[2][4], rz[2][4];
#if MW_SAVE_PG && MW_PG_METHOD==0
      uint32_t pp[2][4],gg[2][4];
#endif
#pragma unroll
      for (int kb=0;kb<2;++kb) {
        ldsm_x4(rz[kb],sZ_u+(uint32_t)(b0>>1)*CHB+
          swz128((uint32_t)(rho0+lrow),(uint32_t)(((2*h+kb)*16+((mat&2)?8:0))*2)));
        zdep ^= rz[kb][0] ^ rz[kb][3];
      }
#pragma unroll
      for (int j=0;j<4;++j) {
        float v[4]={0.f,0.f,0.f,0.f};
#pragma unroll
        for (int r=0;r<2;++r) {
          const int jr=jw+16*wiw+gq+8*r, c=BN*(b0+h)+8*j+q2;
          const float p0=math::round_bf16(accP[4*j+2*r]);
          const float p1=math::round_bf16(accP[4*j+2*r+1]);
          const float g0=math::sigmoid(math::round_bf16(accG[4*j+2*r]));
          const float g1=math::sigmoid(math::round_bf16(accG[4*j+2*r+1]));
#if MW_SAVE_PG
#if MW_PG_METHOD==0
          pp[j>>1][2*(j&1)+r]=pack_bf16(p0,p1);gg[j>>1][2*(j&1)+r]=pack_bf16(g0,g1);
#else
          if(iw<p.N && jr<p.N){stg32(tp.proj+((size_t)iw*p.N+jr)*CZ+c,pack_bf16(p0,p1));stg32(tp.gate+((size_t)iw*p.N+jr)*CZ+c,pack_bf16(g0,g1));}
#endif
#endif
          if (iw<p.N && jr<p.N) {
            const uint32_t ds=ldg32(tp.dropscale+(size_t)jr*CZ+c);
            const uint32_t res=rz[j>>1][2*(j&1)+r];
            v[2*r]=fmaf(p0*g0,bf16lo(ds),bf16lo(res));
            v[2*r+1]=fmaf(p1*g1,bf16hi(ds),bf16hi(res));
          }
        }
        fr[j>>1][2*(j&1)]=pack_bf16(v[0],v[1]);
        fr[j>>1][2*(j&1)+1]=pack_bf16(v[2],v[3]);
      }
#if MW_SAVE_PG && MW_PG_METHOD==0
      const uint32_t ps=smem_u32(sProj)+(4*cw+wiw)*OB,gs=smem_u32(sGate)+(4*cw+wiw)*OB;
      stsm_x4(ps+swz128(lrow,(4*h+(mat>>1))*16),pp[0][0],pp[0][1],pp[0][2],pp[0][3]);stsm_x4(ps+swz128(lrow,(4*h+2+(mat>>1))*16),pp[1][0],pp[1][1],pp[1][2],pp[1][3]);
      stsm_x4(gs+swz128(lrow,(4*h+(mat>>1))*16),gg[0][0],gg[0][1],gg[0][2],gg[0][3]);stsm_x4(gs+swz128(lrow,(4*h+2+(mat>>1))*16),gg[1][0],gg[1][1],gg[1][2],gg[1][3]);
#endif
      stsm_x4(stg_u+swz128((uint32_t)lrow,(uint32_t)((4*h+(mat>>1))*16)),fr[0][0],fr[0][1],fr[0][2],fr[0][3]);
      stsm_x4(stg_u+swz128((uint32_t)lrow,(uint32_t)((4*h+2+(mat>>1))*16)),fr[1][0],fr[1][1],fr[1][2],fr[1][3]);
    };
    // vector pass over the staged pair (blocks b0, b0+1 = 64 contiguous channels): 16-byte granules staging -> (+ residual) -> global, ragged-predicated
    // staged pair (blocks b0, b0+1 = 64 contiguous channels) (+ residual from the z tile still resident in shared memory) -> global, 16-B granules
    auto vector_pass = [&](int b0) {
      __syncwarp();
#ifdef TMN_DEV_NOVEC
      if (lane == 99 && b0 == 12345) sts32(stg_u, 0u);
      return;
#endif
      uint32_t dep = 0;
#pragma unroll
      for (int it = 0; it < NGR; ++it) {
        const int row = (STG32 ? (lane >> 4) : (lane >> 3)) + RSTEP * it, cg = cgl;   // slice-relative row of granule it
        const uint4 ov = STG32 ? lds128(stg_u + (uint32_t)row * 256u + (((uint32_t)cg ^ ((uint32_t)row & 7u)) * 16u)) : lds128(stg_u + swz128((uint32_t)row, (uint32_t)(16 * cg)));
        if (RESG) {                                      // modes 2, 3: out (fp32, 8 values = 32 B) = z (fp32, global) + o (8 bf16 of the granule)
          if (it < nvalid) {
            const size_t boff = it * GSTRIDE + (size_t)(BN * b0 * OSZ);
            uint4 z0 = make_uint4(0u, 0u, 0u, 0u), z1 = z0;
            if (!UPD && p.residual) { z0 = ldg128(zrow + boff); z1 = ldg128(zrow + boff + 16); }
            uint4 w0, w1;
            w0.x = __float_as_uint(math::residual_f32(__uint_as_float(z0.x), bf16lo(ov.x))); w0.y = __float_as_uint(math::residual_f32(__uint_as_float(z0.y), bf16hi(ov.x)));
            w0.z = __float_as_uint(math::residual_f32(__uint_as_float(z0.z), bf16lo(ov.y))); w0.w = __float_as_uint(math::residual_f32(__uint_as_float(z0.w), bf16hi(ov.y)));
            w1.x = __float_as_uint(math::residual_f32(__uint_as_float(z1.x), bf16lo(ov.z))); w1.y = __float_as_uint(math::residual_f32(__uint_as_float(z1.y), bf16hi(ov.z)));
            w1.z = __float_as_uint(math::residual_f32(__uint_as_float(z1.z), bf16lo(ov.w))); w1.w = __float_as_uint(math::residual_f32(__uint_as_float(z1.w), bf16hi(ov.w)));
            stg128(orow + boff, w0); stg128(orow + boff + 16, w1);
          }
          continue;
        }
        uint4 zr = make_uint4(0u, 0u, 0u, 0u);
        if (!UPD && !NORES && p.residual) {                        // z chunk rows are [BMT tok][128 B] 128B-swizzled; bf16: chunk b0/2, granule cg; fp32: chunk b0 + cg/8
          const uint32_t trow = (uint32_t)(tok0 + rl + RSTEP * it);
          zr = ZF32 ? lds128(sZ_u + (uint32_t)(b0 + (cg >> 3)) * (uint32_t)CHB + swz128(trow, (uint32_t)(16 * (cg & 7))))
                    : lds128(sZ_u + (uint32_t)(b0 >> 1) * (uint32_t)CHB + swz128(trow, (uint32_t)(16 * cg)));
          dep ^= zr.x ^ zr.w;
        }
        if (it < nvalid) {
          uint4 w4;
          if (!STG32) {                                   // out = bf16(z + o) (o already bf16; z = 0 without residual): the framework's bf16 residual add
            w4.x = math::residual_bf16x2(zr.x, ov.x); w4.y = math::residual_bf16x2(zr.y, ov.y);
            w4.z = math::residual_bf16x2(zr.z, ov.z); w4.w = math::residual_bf16x2(zr.w, ov.w);
          } else {                                        // out = fp32(z) + o
            w4.x = __float_as_uint(math::residual_f32(__uint_as_float(zr.x), __uint_as_float(ov.x)));
            w4.y = __float_as_uint(math::residual_f32(__uint_as_float(zr.y), __uint_as_float(ov.y)));
            w4.z = __float_as_uint(math::residual_f32(__uint_as_float(zr.z), __uint_as_float(ov.z)));
            w4.w = __float_as_uint(math::residual_f32(__uint_as_float(zr.w), __uint_as_float(ov.w)));
          }
          stg128(orow + it * GSTRIDE + (size_t)(BN * b0 * OSZ), w4);
        }
      }
      if (!NORES && p.residual && !RESG) {                         // this warp's last use of the pair's z chunk(s): release for the next tile's refill
        dep = zero_dep(__reduce_or_sync(0xffffffffu, dep));
        fence_proxy_async();
        if (lane == 0) {
          if (ZF32) { mbar_arrive_dep(barZ_empty + b0, dep); mbar_arrive_dep(barZ_empty + b0 + 1, dep); }
          else mbar_arrive_dep(barZ_empty + (b0 >> 1), dep);
        }
      }
      __syncwarp();                                      // the slice is free for the next pair's staging
    };
    // TMA store of the staged pair: every lane fences its stmatrix writes towards the async proxy, lane 0 issues one bulk-tensor store of the slice
    // (box [64 ch][16 tok][1] at channel BN b0, this warp's token rows; the tensor bounds clip the ragged edge) and releases the pair's z chunk.
    auto store_pass = [&](int b0) {
      const uint32_t d = zero_dep(__reduce_or_sync(0xffffffffu, zdep));
      fence_proxy_async();
      __syncwarp();
      if (lane == 0) {
#if MW_SAVE_PG && MW_PG_METHOD==0
        tma_store_3d(&tp.tm_proj,sProj+(4*cw+wiw)*OB,BN*b0,jw+16*wiw,iw);
        tma_store_3d(&tp.tm_gate,sGate+(4*cw+wiw)*OB,BN*b0,jw+16*wiw,iw);
#endif
        tma_store_3d(&p.tm_out, sOut + (4 * cw + wiw) * OB, BN * b0, jw + 16 * wiw, iw);
        tma_store_commit();
        if (p.residual) mbar_arrive_dep(barZ_empty + (b0 >> 1), d);
      }
      zdep = 0;
    };
    auto out_pass = [&](int b0) { if (K3ST) store_pass(b0); else vector_pass(b0); };
    auto slice_ready = [&]() { if (K3ST) { if (lane == 0) tma_store_wait_read<0>(); __syncwarp(); } };   // the previous store has read the slice
    if (G::NACC == 1) {
#pragma unroll 1
      for (int P = 0; P < NB / 2; ++P) {                 // block pairs in the producer's order; split-N: the pairs alternate between the warpgroups and a
        const int pq = SPLITN ? (P >> 1) * 2 : 2 * P;    // warpgroup passes over the other one's pair (waits + releases its slots without reading them)
        if (SPLITN && (P & 1) != cw) {
          if (!G::W_RESIDENT) {
#pragma unroll
            for (int u = 0; u < 4; ++u) {                // gate, projection slots of blocks 2P, 2P+1
              const uint32_t sq = ((uint32_t)t_local * NB + (uint32_t)(2 * P + (u >> 1))) * 2u + (uint32_t)(u & 1);
              mbar_wait(barW_full + slot_of(sq), phase_of(sq));
            }
            __syncwarp();
            if (lane == 0) {
#pragma unroll
              for (int u = 0; u < 4; ++u) {
                const uint32_t sq = ((uint32_t)t_local * NB + (uint32_t)(2 * P + (u >> 1))) * 2u + (uint32_t)(u & 1);
                mbar_arrive(barW_empty + slot_of(sq));
              }
            }
          }
          continue;
        }
        PF(9);
        const int b0c = blk_of(pq);
        issue(accP0, accG0, pq); wgmma_wait<0>(); PF(10); release(pq); slice_ready(); stage(accP0, accG0, 0, b0c); PF(11);   // 10: MMA wait, 11: stage+release
        issue(accP0, accG0, pq + 1); wgmma_wait<0>(); PF(10); release(pq + 1); stage(accP0, accG0, 1, b0c); PF(11);
        out_pass(b0c); PF(9);                          // 9: vector pass / TMA store
      }
      (void)accP1; (void)accG1;
    } else {                                             // one block's MMAs stay in flight during every staging / vector step
      issue(accP0, accG0, 0); PF(9);
#pragma unroll 1
      for (int pq = 0; pq < NBW; pq += 2) {
        const int b0c = blk_of(pq);
        issue(accP1, accG1, pq + 1);
        wgmma_wait<2>(); PF(10); release(pq); slice_ready(); stage(accP0, accG0, 0, b0c); PF(11);
        if (pq + 2 < NBW) { issue(accP0, accG0, pq + 2); wgmma_wait<2>(); } else { wgmma_wait<0>(); }
        PF(10); release(pq + 1); stage(accP1, accG1, 1, b0c); PF(11);
        out_pass(b0c); PF(9);
      }
    }
  }
  if (K3ST && lane == 0) tma_store_wait_all();          // every bulk store of this warp complete before the CTA's shared memory is released
#ifdef TMN_DEV_PROF
  if (lane == 0 && p.prof) { for (int i = 0; i < 12; ++i) atomicAdd(p.prof + i, pf[i]); atomicAdd(p.prof + 12, 1ull); }
#endif
}}}
extern "C" __global__ __launch_bounds__(384,1)
void save_k3(__grid_constant__ const tmn::sm90::RecomputeParams p){tmn::sm90::infer_k3_body<tmn::sm90::Cfg,1>(p);}
