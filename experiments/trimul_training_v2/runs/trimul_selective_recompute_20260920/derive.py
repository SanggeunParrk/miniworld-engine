"""Minimal rematerialization kernels derived from the same Anthropic saves.

K1: input LN/stats and raw gate/projection only. No sigmoid/mask/ab output.
K3: output LN/stats and projection/sigmoid only. No residual/dropout/y output.
Original independent development files remain untouched.
"""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
front=R.parent/'anthropic_ln_equal_saves_20260919/front.cu'
output=R.parent/'trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/trimul_inproj/cuda/anthropic_k3_training.cu'
s=front.read_text()
s=s.replace('FrontBase::SMEM+2*FrontBase::SMEM_STAGE','FrontBase::SMEM+FrontBase::SMEM_STAGE')
s=s.replace('uint8_t* sGate = sStage + G::SMEM_STAGE;','uint8_t* sGate = sStage;')
s=s.replace('    if (TMN_K1_BULK_STORE) tma_prefetch_desc(&p.tm_ab);','')
a=s.index('    if (HAS_MASK) {');b=s.index('    const int rho_g',a);s=s[:a]+s[b:]
s=s.replace('uint32_t pk[4][2], pg[4][2], pp[4][2];','uint32_t pg[4][2], pp[4][2];')
a=s.index('        // Match existing front rounding:');b=s.index('\n      }',a);s=s[:a]+s[b:]
a=s.index('      const uint32_t sbuf');b=s.index('      const uint32_t gbuf',a);s=s[:a]+s[b:]
a=s.index('      if (TMAST && p.vec) {');b=s.index('\n    };',a)
s=s[:a]+'''      static_assert(TMAST,"Rematerialization covers contiguous 64-token tiles");
      fence_proxy_async();
      if (st_elect) tma_store_wait_read<0>();
      named_bar_sync(bar_id,128);
      if (st_elect) {
        tma_store_3d(&tp.tm_gate,sGate+cw*8192+(b&1)*4096,jw,iw,32*b);
        tma_store_3d(&tp.tm_proj,sProj+cw*8192+(b&1)*4096,jw,iw,32*b);
        tma_store_commit();
      }'''+s[b:]
s=s.replace('void mw_saved_front(', 'void mw_recompute_front(')
s='// Selective recomputation: no a/b output, no gated/masked epilogue.\n'+s
(R/'recompute_front.cu').write_text(s)
s=output.read_text()
s=s.replace('::SMEM + 2 * 8 * 2048','::SMEM + 8 * 2048')
s=s.replace('uint8_t* sProj = sOut + G::SMEM_OUT;','uint8_t* sProj = sOut;')
s=s.replace('    tma_prefetch_desc(&p.tm_out);','')
s=s.replace(' tma_prefetch_desc(&tp.tm_residual);','')
s=s.replace('if (t_local > 0) mbar_wait(barZ_empty + kc, 1);','if (t_local > 0) mbar_wait(barZ_empty + kc, (t_local-1)&1);')
a=s.index('        // Reuse the Z stage');b=s.index('\n      }\n    } else if (warp == 1',a);s=s[:a]+s[b:]
s=s.replace('mbar_wait(barZ_full + kc, 0);','mbar_wait(barZ_full + kc, t_local&1);')
a=s.index('    // Split-N: both warpgroups');b=s.index('    PF(5);',a);s=s[:a]+s[b:]
a=s.index('    // Training rounding contract');b=s.index('    auto slice_ready',a)
s=s[:a]+'''    // Recreate only projection and sigmoid gate consumed by B1-B4.
    auto stage = [&](float (&accP)[16],float (&accG)[16],int h,int b0) {
      fence_regs(accP);fence_regs(accG);
      uint32_t fp[2][4],fg[2][4];
#pragma unroll
      for(int j=0;j<4;++j) {
#pragma unroll
        for(int r=0;r<2;++r) {
          fp[j>>1][2*(j&1)+r]=pack_bf16(accP[4*j+2*r],accP[4*j+2*r+1]);
          fg[j>>1][2*(j&1)+r]=pack_bf16(math::sigmoid(math::round_bf16(accG[4*j+2*r])),math::sigmoid(math::round_bf16(accG[4*j+2*r+1])));
        }
      }
#pragma unroll
      for(int kb=0;kb<2;++kb) {
        const uint32_t off=swz128((uint32_t)lrow,(uint32_t)((4*h+2*kb+(mat>>1))*16));
        stsm_x4(proj_u+off,fp[kb][0],fp[kb][1],fp[kb][2],fp[kb][3]);
        stsm_x4(gate_u+off,fg[kb][0],fg[kb][1],fg[kb][2],fg[kb][3]);
      }
    };
    auto out_pass = [&](int b0) {
      fence_proxy_async();__syncwarp();
      if(lane==0) {
        tma_store_3d(&tp.tm_proj,sProj+(4*cw+wiw)*OB,BN*b0,jw+16*wiw,iw);
        tma_store_3d(&tp.tm_gate,sGate+(4*cw+wiw)*OB,BN*b0,jw+16*wiw,iw);
        tma_store_commit();
      }
    };
'''+s[b:]
s=s.replace('void mw_k3_train(', 'void mw_recompute_output(')
s='// Selective recomputation: no residual/dropout/y, one normalized-Z load per tile.\n'+s
(R/'recompute_output.cu').write_text(s)
(R/'derivation.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (front,output,R/'recompute_front.cu',R/'recompute_output.cu')},indent=2))
