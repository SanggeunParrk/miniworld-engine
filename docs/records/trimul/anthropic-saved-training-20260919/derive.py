from pathlib import Path
E=Path('/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/engine')
D=E/'src/miniworld_engine/kernels/trimul_inproj/cuda'
u=(E/'third_party/anthropic/upstream/common/opt_core/opt_core/kernels/trimul/native/pkg/v5/csrc/tmn_kernels.cuh').read_text()
a=u.index('template <class G, bool HAS_MASK, int LNM, bool SAVE, bool EMITX = false>\nTMN_DEVI void k1_body')
b=u.index('// ============================================================================================================ K3',a)
s=u[a:b]
s=s.replace('void k1_body(const K1Params& p) {','void saved_front_body(const SavedFrontParams& tp) {\n  const auto& p = tp.base;')
s=s.replace('float* sGamma = reinterpret_cast<float*>(sStage + G::SMEM_STAGE);','uint8_t* sGate = sStage + G::SMEM_STAGE;\n  uint8_t* sProj = sGate + G::SMEM_STAGE;\n  float* sGamma = reinterpret_cast<float*>(sProj + G::SMEM_STAGE);')
s=s.replace('for (int i = tid; i < CZ; i += G::NTHR) { sGamma[i] = p.gamma[i]; sBeta[i] = p.beta[i]; }','// Prenormalized x_n: input LN and its saves stay in their existing kernel.')
s=s.replace('tma_prefetch_desc(&p.tm_ab);','tma_prefetch_desc(&p.tm_ab);\n    tma_prefetch_desc(&tp.tm_gate); tma_prefetch_desc(&tp.tm_proj);')
start=s.index('      uint32_t pk[4][2];')
end=s.index('      const uint32_t sbuf =',start)
s=s[:start]+'''      // The preactivation layout is unchanged: [g0,p0,g1,p1,...] x M.
      // Stage gate/projection separately and TMA-store through stride-2-channel maps.
      uint32_t pk[4][2], pg[4][2], pp[4][2];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        pg[q][0] = pack_bf16(ac[4*q], ac[4*q+1]);
        pg[q][1] = pack_bf16(ac[4*q+2], ac[4*q+3]);
        pp[q][0] = pack_bf16(ac[4*(q+4)], ac[4*(q+4)+1]);
        pp[q][1] = pack_bf16(ac[4*(q+4)+2], ac[4*(q+4)+3]);
        // Match existing front rounding: gate FP32 accumulators, BF16 before mask.
        float vA0 = math::round_bf16(math::gate(ac[4*q],ac[4*(q+4)])) * mA;
        float vA1 = math::round_bf16(math::gate(ac[4*q+1],ac[4*(q+4)+1])) * mA;
        float vB0 = math::round_bf16(math::gate(ac[4*q+2],ac[4*(q+4)+2])) * mB;
        float vB1 = math::round_bf16(math::gate(ac[4*q+3],ac[4*(q+4)+3])) * mB;
        if (!vA) { vA0=0.f; vA1=0.f; }
        if (!vB) { vB0=0.f; vB1=0.f; }
        pk[q][0]=pack_bf16(vA0,vA1); pk[q][1]=pack_bf16(vB0,vB1);
      }
''' +s[end:]
pos=s.index('      if (TMAST && p.vec) {')
s=s[:pos]+'''      const uint32_t gbuf=smem_u32(sGate)+(uint32_t)(cw*8192+(b&1)*4096);
      const uint32_t pbuf=smem_u32(sProj)+(uint32_t)(cw*8192+(b&1)*4096);
      stsm_x4_t(gbuf+sts_off0,pg[0][0],pg[0][1],pg[1][0],pg[1][1]);
      stsm_x4_t(gbuf+sts_off1,pg[2][0],pg[2][1],pg[3][0],pg[3][1]);
      stsm_x4_t(pbuf+sts_off0,pp[0][0],pp[0][1],pp[1][0],pp[1][1]);
      stsm_x4_t(pbuf+sts_off1,pp[2][0],pp[2][1],pp[3][0],pp[3][1]);
''' +s[pos:]
s=s.replace('tma_store_commit(); }','''
          tma_store_3d(&tp.tm_gate,sGate+cw*8192+(b&1)*4096,jw,iw,32*b);
          tma_store_3d(&tp.tm_proj,sProj+cw*8192+(b&1)*4096,jw,iw,32*b);
          tma_store_commit(); }''')
header='''// SPDX-License-Identifier: Apache-2.0
// Derived from Anthropic uplifting-biomolecular-modeling f4f62fa6592ae4938d49b1757bea0cfeff9f468e,
// native/pkg/v5/csrc/tmn_kernels.cuh (k1_body). Original vendored source unchanged.
// Changes: consume existing normalized input (LNM=0), preserve original training
// preactivation saves/rounding, TMA-store interleaved gate/projection as well as a/b.
#include "tmn_kernels.cuh"
namespace tmn { namespace sm90 {
struct SavedFrontParams { K1Params base; CUtensorMap tm_gate,tm_proj; };
static_assert(sizeof(K1Params)==512,"upstream K1 ABI");
static_assert(sizeof(SavedFrontParams)==768,"saved front ABI");
using FrontBase=K1Cfg<MWK1_CZ,MWK1_CH,false,MWK1_BI,MWK1_BJ,MWK1_NSLOT,MWK1_SKCH,MWK1_SCHED>;
struct SavedFrontCfg : FrontBase {
  static constexpr int SMEM=FrontBase::SMEM+2*FrontBase::SMEM_STAGE;
  static_assert(SMEM*MINB<=SMEM_LIMIT,"training saves exceed shared memory");
};
'''
footer='''
}}
extern "C" __global__ __launch_bounds__(tmn::sm90::SavedFrontCfg::NTHR,tmn::sm90::SavedFrontCfg::MINB)
void mw_saved_front(__grid_constant__ const tmn::sm90::SavedFrontParams p) {
 tmn::sm90::saved_front_body<tmn::sm90::SavedFrontCfg,true,0,false>(p);
}
'''
(D/'anthropic_saved_front.cu').write_text(header+s+footer)
s=(D/'anthropic_k3_training.cu').read_text()
s=s.replace('// MiniWorld changes: prenormalized input, output LN saves, projection/gate saves,','// MiniWorld saved-policy variant: both operands already normalized by existing LN kernels.\n// No LN fusion/recomputation here; keep projection/gate saves and dropout/residual.\n// MiniWorld changes: prenormalized input, projection/gate saves,')
s=s.replace('TrainParams','SavedOutputParams').replace('TrainCfg','SavedOutputCfg').replace('k3_training_body','saved_output_body').replace('mw_k3_train','mw_saved_output')
# Remove normalization parameter loads (unused); retain layout to minimize ABI changes.
s=s.replace('for (int i = tid; i < CZ; i += NTHREADS) { sGin[i] = p.gamma_in[i]; sBin[i] = p.beta_in[i]; }','')
s=s.replace('for (int i = tid; i < CH; i += NTHREADS) { sGout[i] = p.gamma_out[i]; sBout[i] = p.beta_out[i]; }','')
s=s.replace('tma_prefetch_desc(&tp.tm_norm); ','')
a=s.index('#pragma unroll\n        for (int h = 0; h < G::NSUB; ++h)')
b=s.index('#pragma unroll 1\n        for (int kc = 0; kc < NKCZ;',a)
s=s[:a]+'''#pragma unroll
        for (int kc=0;kc<CH/64;++kc)
          tma_load_3d(sX+kc*(BI*BJ*128),&p.tm_x,barX_full,kc*64,j0,i0);
'''+s[b:]
s=s.replace('smem_u32(sX) + (uint32_t)((tok0 / 64) * (CH * 128))','smem_u32(sX)')
a=s.index('      const int tokc = 16 * wiw')
b=s.index('      uint32_t dep = 0;',a)
s=s[:a]+'''      load_frag_bf16<KSP, BI*BJ*128>(fx,sX_u,rho0,lane);
'''+s[b:]
a=s.index('    const LnStats stats =')
b=s.index('    PF(3);',a)
s=s[:a]+'''    // F4 already produced/stored normalized X and mean/rstd. No LN here.
'''+s[b:]
(D/'anthropic_saved_output.cu').write_text(s)
print('Derived saved-policy front and F567 sources')
