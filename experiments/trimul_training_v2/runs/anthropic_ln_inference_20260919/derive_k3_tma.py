from pathlib import Path
R=Path(__file__).resolve().parent;E=R.parent/'trimul_sm90_parity_20260917/engine'
s=(E/'src/miniworld_engine/kernels/trimul_inproj/cuda/anthropic_k3_training.cu').read_text()
a=s.index('struct TrainParams {');b=s.index('struct TrainCfg',a)
s=s[:a]+'struct TrainParams { K3Params base; CUtensorMap tm_residual; };\n'+s[b:]
s=s.replace(' + 2 * 8 * 2048','')
s=s.replace('  uint8_t* sProj = sOut + G::SMEM_OUT;\n  uint8_t* sGate = sProj + G::SMEM_OUT;\n  float* sGin = reinterpret_cast<float*>(sGate + G::SMEM_OUT);','  float* sGin = reinterpret_cast<float*>(sOut + G::SMEM_OUT);')
s='\n'.join(line for line in s.splitlines() if 'const uint32_t proj_u' not in line and 'const uint32_t gate_u' not in line)
s=s.replace('    tma_prefetch_desc(&tp.tm_norm); tma_prefetch_desc(&tp.tm_proj); tma_prefetch_desc(&tp.tm_gate); tma_prefetch_desc(&tp.tm_residual);','    tma_prefetch_desc(&tp.tm_residual);')
s=s.replace('for (int i = tid; i < CZ; i += NTHREADS) { sGin[i] = p.gamma_in[i]; sBin[i] = p.beta_in[i]; }','')
a=s.index('    const LnStats stats =');b=s.index('    PF(3);',a)
s=s[:a]+'    ln_fragment<KSP,MW_SERIAL>(fx,sGout,sBout,lane,p.eps);\n'+s[b:]
u=(E/'third_party/anthropic/upstream/common/opt_core/opt_core/kernels/trimul/native/pkg/v5/csrc/tmn_kernels.cuh').read_text();a=u.index('    auto stage =',u.index('TMN_DEVI void k3_body'));b=u.index('    // vector pass',a);stage=u[a:b]
stage=stage.replace('      fence_regs(accP); fence_regs(accG);','      mbar_wait(barZ_full+(b0>>1),1);\n      fence_regs(accP); fence_regs(accG);')
a=s.index('    // Training rounding contract');b=s.index('    // vector pass',a);s=s[:a]+stage+s[b:]
s='\n'.join(line for line in s.splitlines() if 'tma_store_3d(&tp.tm_proj' not in line and 'tma_store_3d(&tp.tm_gate' not in line)
s=s.replace('const bool now = !(p.residual && mine && !RESG && !NORES);','const bool now = true;')
s=s.replace('mw_k3_train','infer_k3_tma')
macros='''// Inference only: no training saves/dropout; upstream inference rounding.
#define MWK3_CZ 128
#define MWK3_CH 256
#define MWK3_BI MW_BI
#define MWK3_BJ MW_BJ
#define MWK3_NSLOT MW_SLOT
#define MWK3_NACC MW_ACC
#define MWK3_REGS (TMN_K3_REGS_24_240 ? 240 : 232)
#define MWK3_LNSERIAL MW_SERIAL
'''
(R/'k3_tma.cu').write_text(macros+s+'\n')
