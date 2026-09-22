from pathlib import Path
R=Path(__file__).resolve().parent/'dnreg';P=R.parent/'pair';R.mkdir(exist_ok=True)
for p in [*P.glob('*.cuh'),*P.glob('*.inc'),P/'b1_fused.cu',P/'replace_plan.py']:(R/p.name).write_bytes(p.read_bytes())
p=R/'b1_fused.cu';p.write_text(p.read_text().replace('#include <cuda_fp16.h>','#include <cuda_fp16.h>\n#ifndef B1_DN_REG\n#define B1_DN_REG 0\n#endif'))
p=R/'lowreg_stats.inc';s=p.read_text().replace(' float* stats=reinterpret_cast<float*>(sm+224256);',' uint32_t dnreg[2][4][4];\n float* stats=reinterpret_cast<float*>(sm+224256);')
lines=s.splitlines();out=[]
for line in lines:
 if line.strip().startswith('stsm_x4(smem_u32((B1_SPLIT_DN'):
  out+=['#if B1_DN_REG','   static_for<4>([&](auto ji){constexpr int j=decltype(ji)::value;dnreg[nl][q][j]=dn[j];});','#else',line,'#endif']
 elif line.strip().startswith('ldsm_x4(dn,smem_u32((B1_SPLIT_DN'):
  out+=['#if B1_DN_REG','   static_for<4>([&](auto ji){constexpr int j=decltype(ji)::value;dn[j]=dnreg[nl][q][j];});','#else',line,'#endif']
 else:out.append(line)
p.write_text('\n'.join(out)+'\n')
