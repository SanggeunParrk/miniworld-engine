from pathlib import Path
R=Path(__file__).resolve().parent
D=R.parent/'trimul_b7_gate_half_20260922';D.mkdir(exist_ok=True)
for name in ('single_wg.inc','precompile.py','probe.py','ncu_profile.py','sanitize.py','run.slurm','sanitize.slurm','profile.slurm','sweep.py'):
 (D/name).write_text((R/name).read_text().replace(R.name,D.name))
p=(R/'plan.py').read_text().replace("flags=['-DB7_PACK_DX=", "flags=['-DB7_HALF_GATE='+str((int(os.environ.get('B7_PACK','0'))>>2)&1),'-DB7_PACK_DX=")
(D/'plan.py').write_text(p)
s=(R/'joint.cu').read_text()
old='{float gate[64]={};uint32_t unused[8][4];mbar_wait(bar+19,round&1);wide_gate(gate,unused,sm+WGATE,sm+98304);static_for<64>([&](auto jj){constexpr int j=decltype(jj)::value;acc[j]=math::round_bf16(acc[j]+math::round_bf16(gate[j]));});}'
assert s.count(old)==1
new='''#if B7_HALF_GATE
  mbar_wait(bar+19,round&1);
  static_for<2>([&](auto hh){constexpr int h=decltype(hh)::value;float gate[32]={};fence_regs(gate);wgmma_fence();
   static_for<8>([&](auto kk){constexpr int k=decltype(kk)::value;
    gp_ss64(gate,smem_desc(smem_u32(sm+98304+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sm+WGATE+(k/4)*16384+h*8192+(k%4)*32),16,1024,1),k>0);
   });wgmma_commit();wgmma_wait<0>();fence_regs(gate);
   static_for<32>([&](auto jj){constexpr int j=decltype(jj)::value;acc[h*32+j]=math::round_bf16(acc[h*32+j]+math::round_bf16(gate[j]));});
  });
#else
  '''+old+'''
#endif'''
s=s.replace(old,new)
(D/'joint.cu').write_text(s)

E=R.parent/'trimul_b7_384_control_20260922';E.mkdir(exist_ok=True)
for name in ('single_wg.inc','precompile.py','probe.py','ncu_profile.py','sanitize.py','run.slurm','sanitize.slurm','profile.slurm','sweep.py'):
 (E/name).write_text((D/name).read_text().replace(D.name,E.name))
p=p.replace('cfg.blockDimX=256','cfg.blockDimX=384').replace('self.count,1,1,256,1,1','self.count,1,1,384,1,1')
(E/'plan.py').write_text(p)
s=s.replace('THREADS=256','THREADS=384').replace('__launch_bounds__(256,2)','__launch_bounds__(384,2)')
s=s.replace('if(rank<SOURCES){\n  if(threadIdx.x==0)', 'if(wi==2){setmaxnreg_dec<24>();}\n else if(rank<SOURCES){\n  if(threadIdx.x==0)')
# Startup source __syncthreads cannot be inside a branch excluding WG 2.
s=s.replace('mbar_wait(bar+18,0);__syncthreads();','mbar_wait(bar+18,0);named_bar_sync(14,256);')
s=s.replace('setmaxnreg_inc<256-B7_PRODUCER_REGS>()','setmaxnreg_inc<216-B7_PRODUCER_REGS>()')
(E/'joint.cu').write_text(s)
print(D,E)
