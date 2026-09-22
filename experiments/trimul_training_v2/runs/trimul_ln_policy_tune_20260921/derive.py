from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent/'trimul_replace_tri_20260921';BASE=R.parent/'trimul_b1_shared_20260921'
for p in [*P.glob('*.cuh'),*P.glob('*.inc'),P/'b1_fused.cu',P/'replace_plan.py',P/'replace_core.py',P/'save_k3.cu',P/'bench.py']:(R/p.name).write_bytes(p.read_bytes())
for name in ('save_k3.cu','b1_fused.cu','b1_lowreg.inc','xhat_read.inc'):
 p=R/name;s=p.read_text().replace('#if XHAT_FP32','#if XHAT_FP32 == 1');p.write_text(s)
s=(R/'save_k3.cu').read_text().replace('#include "tmn_kernels.cuh"','#include "tmn_kernels.cuh"\n#include <cuda_fp16.h>')
s=s.replace('    // Save pre-affine normalization','''#if XHAT_FP32 >= 0
    // Save pre-affine normalization''').replace('    const uint32_t k0 =','''#endif
    const uint32_t k0 =''')
s=s.replace('#else\n     reinterpret_cast<__nv_bfloat16*>', '''#elif XHAT_FP32 >= 2
     reinterpret_cast<__half*>(output)[(size_t)cc*M+rr]=__float2half_rn(a);reinterpret_cast<__half*>(output)[(size_t)(cc+1)*M+rr]=__float2half_rn(b);
#else
     reinterpret_cast<__nv_bfloat16*>''')
s=s.replace('tp.rs_out[(size_t)iw*p.N+r]=stats_out.rA;', 'tp.rs_out[(size_t)iw*p.N+r]=stats_out.rA;\n#if XHAT_FP32 == -1 || XHAT_FP32 == 3\n tp.mean_out[(size_t)iw*p.N+r]=stats_out.mA;\n#endif\n')
s=s.replace('tp.rs_out[(size_t)iw*p.N+r+8]=stats_out.rB;', 'tp.rs_out[(size_t)iw*p.N+r+8]=stats_out.rB;\n#if XHAT_FP32 == -1 || XHAT_FP32 == 3\n tp.mean_out[(size_t)iw*p.N+r+8]=stats_out.mB;\n#endif\n')
(R/'save_k3.cu').write_text(s)
s=(R/'replace_core.py').read_text().replace('dtype=torch.float32 if method else x.dtype','dtype=torch.float32 if method==1 else torch.float16 if method>=2 else x.dtype')
s=s.replace('mo=None,ro=f() if stats&2 else None','mo=None,ro=torch.empty(m*(2 if method in (-1,3) else 1),device=x.device,dtype=torch.float32) if stats&2 else None')
s=s.replace(" else:y,saves=bufs\n maps=", " else:y,saves=bufs\n if method in (-1,3):saves['mo']=saves['ro'][m:]\n if method==-1:saves['xnout']=tri\n maps=")
# Avoid allocating the unused output in stats-only mode.
s=s.replace("xnout=torch.empty", "xnout=tri if method==-1 else torch.empty")
(R/'replace_core.py').write_text(s)
s=(R/'xhat_read.inc').read_text().replace('TMN_DEVI float xhat_at','TMN_DEVI float xhat_at')
s=s.replace('#else\n return __bfloat162float', '''#elif XHAT_FP32 >= 2
 float a=__half2float(*reinterpret_cast<__half*>(sm+32768+slot*49152+swz128(c,row*2)));
#if XHAT_FP32 == 3
 float* mu=reinterpret_cast<float*>(sm+225280);float rs=mu[64+row];
 float raw=math::round_bf16(__fadd_rn(__fdiv_rn(a,rs),mu[row]));
 return __fmul_rn(__fsub_rn(raw,mu[row]),rs);
#else
 return a;
#endif
#else
 return __bfloat162float''')
(R/'xhat_read.inc').write_text(s)
s=(R/'b1_fused.cu').read_text().replace('#include "common_recompute.cuh"','#include "common_recompute.cuh"\n#include <cuda_fp16.h>')
s=s.replace('#include "b1_lowreg.inc"','#if XHAT_FP32 == -1\n#include "lowreg_stats.inc"\n#else\n#include "b1_lowreg.inc"\n#endif')
s=s.replace('rs[rb]=p.saved_rs[row+rb];}', '''rs[rb]=p.saved_rs[row+rb];
#if XHAT_FP32 == -1 || XHAT_FP32 == 3
 rs[ra-64]=p.saved_rs[p.M+row+ra];rs[rb-64]=p.saved_rs[p.M+row+rb];
#endif
 }
 __syncwarp();''')
a=s.index('  #pragma unroll 4\n  for(int k=0;k<16;++k)');b=s.index('\n }\n fence_proxy_async();allsync();',a)
affine=s[a:b]
original=(BASE/'b1_stream_ln.inc').read_text();a0=original.index(' constexpr int affine_unroll');b0=original.index(' return LnStats',a0)
oldaffine=original[a0:b0]
oldaffine=oldaffine.replace('B1_STREAM_AFFINE_UNROLL','4').replace('src','x+16384').replace('dst','norm').replace('gamma','(ps+256)').replace('beta','(ps+512)')
# raw tile matrix fragment uses lane%8 r8
oldaffine=' int r8=lane%8;float* mus=reinterpret_cast<float*>(sm+225280);float ma=mus[ra],mb=mus[rb],rA=mus[64+ra],rB=mus[64+rb];\n'+oldaffine.replace(',ma,ra,',',ma,rA,').replace(',mb,rb,',',mb,rB,')
s=s[:a]+'#if XHAT_FP32 == -1\n'+oldaffine+'#else\n'+affine+'\n#endif'+s[b:]
(R/'b1_fused.cu').write_text(s);(R/'lowreg_stats.inc').write_bytes((BASE/'b1_lowreg.inc').read_bytes())
# clone benchmark, preserve inherited class and use several candidate formats.
s=(R/'bench.py').read_text().replace("'xhat_bf16':Replacement(a,False),'xhat_fp32':Replacement(a,True)","'stats_only':Replacement(a,-1),'xhat_fp32':Replacement(a,1),'xhat_fp16':Replacement(a,2),'xhat_fp16_recover':Replacement(a,3)")
s=s.replace("if name!='baseline':", "if name not in ('baseline','stats_only'):")
s=s.replace("raw_tri_in_backward=False", "raw_tri_in_backward={'stats_only':True,'xhat_fp32':False,'xhat_fp16':False,'xhat_fp16_recover':False}")
(R/'bench.py').write_text(s)
s=(P/'run.sbatch').read_text().replace('trimul_replace_tri_20260921','trimul_ln_policy_tune_20260921').replace('--array=0-1%1','--array=0-1%2').replace('tri-replace','ln-policy-tune');(R/'run.sbatch').write_text(s)
print('generated')
