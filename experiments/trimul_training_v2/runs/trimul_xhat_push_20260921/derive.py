from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent/'trimul_ln_policy_v5_20260921'
for p in [*P.glob('*.cuh'),*P.glob('*.inc'),P/'b1_fused.cu',P/'replace_plan.py',P/'replace_core.py',P/'save_k3.cu',P/'bench.py']:(R/p.name).write_bytes(p.read_bytes())
s=(R/'b1_lowreg.inc').read_text().replace(' float* tmp=reinterpret_cast<float*>(sm+216064);',' float* tmp=reinterpret_cast<float*>(sm+216064);\n uint32_t out_cache[32];')
a=s.index('#if XHAT_FP32 == 1\n   #pragma unroll');b=s.index('#else',a)
s=s[:a]+'''#if XHAT_FP32 == 1
   #pragma unroll
   for(int j=0;j<4;++j)out_cache[nl*16+q*4+j]=out[j];
'''+s[b:]
a=s.index(' });allsync();\n int c=threadIdx.x');s=s[:a]+s[a:].replace(' });allsync();\n int c=threadIdx.x',''' });allsync();
#if XHAT_FP32 == 1
 // All dNorm reads have completed: recycle its 32 KiB region for dTri.
 static_for<2>([&](auto ni){constexpr int nl=decltype(ni)::value;int n=wi*2+nl;
  static_for<4>([&](auto qi){constexpr int q=decltype(qi)::value;int k=nl*16+q*4;
   stsm_x4_t(smem_u32(sn)+swz128(n*64+q*16+r8+8*(mat>>1),(w*16+8*(mat&1))*2),out_cache[k],out_cache[k+1],out_cache[k+2],out_cache[k+3]);
  });
 });allsync();
 fence_proxy_async();sync_group();if(tid==0){for(int ch=wi*128;ch<(wi+1)*128;ch+=16)tma_store_3d(&p.dtri,sn+ch*128,m0,ch,0);tma_store_commit();tma_store_wait_all();}sync_group();
#endif
 int c=threadIdx.x''',1)
(R/'b1_lowreg.inc').write_text(s)
# Keep only normalized-value candidates: stats-only policy is not selected here.
s=(R/'bench.py').read_text().replace("'stats_only':Replacement(a,-1),",'')
s=s.replace("'xhat_fp32':Replacement(a,1)","'xhat_fp32':Replacement(a,1)")
# Add exact previous normalized path as a matched control (isolated module globals).
s=s.replace('OLD=BASE.OLD;Q=OLD.Q;LN=OLD.LN','''OLD=BASE.OLD;Q=OLD.Q;LN=OLD.LN
import importlib.util
def load_old(name,file):
 spec=importlib.util.spec_from_file_location(name,file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
PR=R.parent/'trimul_ln_policy_v4_20260921'
OLDN=load_old('prior_xhat_bench',PR/'bench.py');OLDN.RP=load_old('prior_xhat_plan',PR/'replace_plan.py');OLDN.RC=load_old('prior_xhat_core',PR/'replace_core.py')''')
s=s.replace("'baseline':BASE.Training(a),'xhat_fp32':Replacement(a,1)","'baseline':BASE.Training(a),'prior_xhat':OLDN.Replacement(a,1),'xhat_fp32':Replacement(a,1)")
(R/'bench.py').write_text(s)
s=(P/'run.sbatch').read_text().replace('trimul_ln_policy_v5_20260921','trimul_xhat_push_20260921').replace('--array=0-1%2','--array=0-1%1').replace('ln-policy-tune','xhat-only');(R/'run.sbatch').write_text(s)
print('Generated FP32-only reuse candidate')
