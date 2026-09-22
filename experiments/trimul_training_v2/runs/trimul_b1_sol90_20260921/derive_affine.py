from pathlib import Path
P=Path(__file__).resolve().parent;R=P/'affine';R.mkdir(exist_ok=True)
for p in [*P.glob('*.cuh'),*P.glob('*.inc'),P/'b1_fused.cu',P/'replace_plan.py']:(R/p.name).write_bytes(p.read_bytes())
s=(R/'replace_plan.py').read_text().replace("OLD=R.parent/'trimul_split_bwd_20260921'","OLD=R.parent.parent/'trimul_split_bwd_20260921'");(R/'replace_plan.py').write_text(s)
s=(R/'b1_fused.cu').read_text().replace('#include <cuda_fp16.h>','#include <cuda_fp16.h>\n#ifndef B1_AFFINE_BOTH\n#define B1_AFFINE_BOTH 0\n#endif')
s=s.replace('if(wi==1){','if(B1_AFFINE_BOTH || wi==1){')
s=s.replace('for(int k=0;k<16;++k){uint32_t f[4];ldsm_x4_t', 'for(int k=(B1_AFFINE_BOTH?wi*8:0);k<(B1_AFFINE_BOTH?(wi+1)*8:16);++k){uint32_t f[4];ldsm_x4_t')
(R/'b1_fused.cu').write_text(s)
