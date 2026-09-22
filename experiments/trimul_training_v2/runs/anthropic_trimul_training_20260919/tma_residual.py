from pathlib import Path
E=Path('/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/engine')
p=E/'src/miniworld_engine/kernels/trimul_inproj/cuda/anthropic_k3_training.cu'
s=p.read_text().replace('CUtensorMap tm_norm, tm_proj, tm_gate;','CUtensorMap tm_norm, tm_proj, tm_gate, tm_residual;').replace('sizeof(TrainParams) == 1216','sizeof(TrainParams) == 1344')
s=s.replace('tma_prefetch_desc(&tp.tm_gate);','tma_prefetch_desc(&tp.tm_gate); tma_prefetch_desc(&tp.tm_residual);')
s=s.replace('mbar_wait(barZ_empty + kc, (t_local - 1) & 1);','mbar_wait(barZ_empty + kc, 1);')
needle='''          tma_load_3d(sZ + kc * CHB, &p.tm_z, barZ_full + kc, kc * G::CHUNK_CH, j0, i0);
        }'''
assert needle in s
s=s.replace(needle,needle+'''
        // Reuse the Z stage after normalized operands have reached registers.
        // Two barrier phases per tile: 0 = normalized input; 1 = residual.
#pragma unroll 1
        for (int kc=0;kc<NKCZ;++kc) {
          mbar_wait(barZ_empty+kc,0);
          mbar_arrive_expect_tx(barZ_full+kc,CHB);
          tma_load_3d(sZ+kc*CHB,&tp.tm_residual,barZ_full+kc,kc*G::CHUNK_CH,j0,i0);
        }''',1)
s=s.replace('mbar_wait(barZ_full + kc, t_local & 1);','mbar_wait(barZ_full + kc, 0);')
needle='''      uint32_t fr[2][4], fp[2][4], fg[2][4];'''
assert needle in s
s=s.replace(needle,needle+'''
      uint32_t rz[2][4];
      mbar_wait(barZ_full+(b0>>1),1);
#pragma unroll
      for (int kb=0;kb<2;++kb) {
        ldsm_x4(rz[kb],sZ_u+(uint32_t)(b0>>1)*CHB+
          swz128((uint32_t)(rho0+lrow),(uint32_t)(((2*h+kb)*16+((mat&2)?8:0))*2)));
        zdep ^= rz[kb][0] ^ rz[kb][3];
      }''',1)
s=s.replace('const uint32_t res = ldg32(reinterpret_cast<const __nv_bfloat16*>(p.zres) + off);','const uint32_t res = rz[j>>1][2*(j&1)+r];')
s=s.replace('if (p.residual) mbar_arrive_dep(barZ_empty + (b0 >> 1), d);','mbar_arrive_dep(barZ_empty + (b0 >> 1), d);')
p.write_text(s)
p=E/'src/miniworld_engine/kernels/trimul_inproj/cuda/anthropic_training.py';s=p.read_text()
s=s.replace("tm(gate,[64,16,1],[cz,n,n],[cz*2,n*cz*2])]", "tm(gate,[64,16,1],[cz,n,n],[cz*2,n*cz*2]),\n               tm(residual,[64,bj,bi],[cz,n,n],[cz*2,n*cz*2])]")
s=s.replace('len(params.pack())==1216','len(params.pack())==1344')
p.write_text(s)
