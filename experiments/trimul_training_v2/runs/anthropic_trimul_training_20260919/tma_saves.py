from pathlib import Path
E=Path('/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/engine')
p=E/'src/miniworld_engine/kernels/trimul_inproj/cuda/anthropic_k3_training.cu'
s=Path('/home/psk6950/MiniWorld/runs/anthropic_trimul_training_20260919/k3-v1.cu').read_text()
s=s.replace('  K3Params base;','  K3Params base;\n  CUtensorMap tm_norm, tm_proj, tm_gate;')
s=s.replace('sizeof(TrainParams) == 832','sizeof(TrainParams) == 1216')
s=s.replace('  static constexpr bool PRENORM = true;','''  static constexpr bool PRENORM = true;
  static constexpr int SMEM = K3Cfg<MWK3_CZ, MWK3_CH, 0, MWK3_BI, MWK3_BJ, MWK3_NSLOT, MWK3_NACC>::SMEM + 2 * 8 * 2048;''')
s=s.replace('  float* sGin = reinterpret_cast<float*>(sOut + G::SMEM_OUT);','''  uint8_t* sProj = sOut + G::SMEM_OUT;
  uint8_t* sGate = sProj + G::SMEM_OUT;
  float* sGin = reinterpret_cast<float*>(sGate + G::SMEM_OUT);''')
s=s.replace('    tma_prefetch_desc(&p.tm_out);','''    tma_prefetch_desc(&p.tm_out);
    tma_prefetch_desc(&tp.tm_norm); tma_prefetch_desc(&tp.tm_proj); tma_prefetch_desc(&tp.tm_gate);''')
s=s.replace('  const int rho0 = tok0 + 16 * wiw;', '''  const uint32_t proj_u = smem_u32(sProj) + (uint32_t)((4*cw+wiw)*OB);
  const uint32_t gate_u = smem_u32(sGate) + (uint32_t)((4*cw+wiw)*OB);
  const int rho0 = tok0 + 16 * wiw;''')
a=s.index('    // Save exactly');b=s.index('    PF(3);',a)
s=s[:a]+'''    // Save normalized WGMMA operands through the same swizzled/TMA path as
    // the upstream output. Reuse projection staging before any MMA output.
    if (!SPLITN || cw == 0) {
      const int ja = jw + 16*wiw + gq, jb = ja + 8;
      if (iw < p.N && (lane & 3) == 0) {
        if (ja < p.N) { tp.mean[iw*p.N+ja]=stats.mA; tp.rstd[iw*p.N+ja]=stats.rA; }
        if (jb < p.N) { tp.mean[iw*p.N+jb]=stats.mB; tp.rstd[iw*p.N+jb]=stats.rB; }
      }
#pragma unroll
      for (int c64=0; c64<CH/64; ++c64) {
        if (lane == 0) tma_store_wait_read<0>();
        __syncwarp();
#pragma unroll
        for (int slab=0; slab<4; ++slab) {
          const int ks=4*c64+slab;
          stsm_x4(proj_u + swz128((uint32_t)lrow,(uint32_t)((2*slab+(mat>>1))*16)),
                  fx[ks][0],fx[ks][1],fx[ks][2],fx[ks][3]);
        }
        fence_proxy_async(); __syncwarp();
        if (lane == 0) {
          tma_store_3d(&tp.tm_norm,sProj+(4*cw+wiw)*OB,64*c64,jw+16*wiw,iw);
          tma_store_commit();
        }
      }
      if (lane == 0) tma_store_wait_read<0>();
      __syncwarp();
    }
''' + s[b:]
s=s.replace('      uint32_t fr[2][4];\n#pragma unroll','      uint32_t fr[2][4], fp[2][4], fg[2][4];\n#pragma unroll',1)
s=s.replace('''            stg32(tp.proj + off, pack_bf16(p0, p1));
            stg32(tp.gate + off, pack_bf16(g0, g1));''','')
s=s.replace('''          if (iw < p.N && jr < p.N) {''','''          fp[j>>1][2*(j&1)+r]=pack_bf16(p0,p1);
          fg[j>>1][2*(j&1)+r]=pack_bf16(g0,g1);
          if (iw < p.N && jr < p.N) {''',1)
needle='''      stsm_x4(stg_u + swz128((uint32_t)lrow, (uint32_t)((4*h+2+(mat>>1))*16)), fr[1][0],fr[1][1],fr[1][2],fr[1][3]);'''
assert needle in s
s=s.replace(needle,needle+'''
#pragma unroll
      for (int kb=0;kb<2;++kb) {
        const uint32_t off=swz128((uint32_t)lrow,(uint32_t)((4*h+2*kb+(mat>>1))*16));
        stsm_x4(proj_u+off,fp[kb][0],fp[kb][1],fp[kb][2],fp[kb][3]);
        stsm_x4(gate_u+off,fg[kb][0],fg[kb][1],fg[kb][2],fg[kb][3]);
      }''')
needle='''        tma_store_3d(&p.tm_out, sOut + (4 * cw + wiw) * OB, BN * b0, jw + 16 * wiw, iw);'''
assert needle in s
s=s.replace(needle,needle+'''
        tma_store_3d(&tp.tm_proj,sProj+(4*cw+wiw)*OB,BN*b0,jw+16*wiw,iw);
        tma_store_3d(&tp.tm_gate,sGate+(4*cw+wiw)*OB,BN*b0,jw+16*wiw,iw);''')
p.write_text(s)
p=E/'src/miniworld_engine/kernels/trimul_inproj/cuda/anthropic_training.py';s=p.read_text()
s=s.replace('smem+=8*2048+', 'smem+=3*8*2048+')
s=s.replace('''params=L.Struct([base,norm,proj,gate,mean,rs,dropscale])
        assert len(base.pack())==768 and len(params.pack())==832''','''saves=[tm(norm,[64,16,1],[ch,n,n],[ch*2,n*ch*2]),
               tm(proj,[64,16,1],[cz,n,n],[cz*2,n*cz*2]),
               tm(gate,[64,16,1],[cz,n,n],[cz*2,n*cz*2])]
        params=L.Struct([base,*saves,norm,proj,gate,mean,rs,dropscale])
        assert len(base.pack())==768 and len(params.pack())==1216''')
p.write_text(s)
