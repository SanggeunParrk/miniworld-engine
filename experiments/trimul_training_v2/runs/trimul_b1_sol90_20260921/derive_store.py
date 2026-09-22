from pathlib import Path
R=Path(__file__).resolve().parent/'store';P=R.parent/'affine';R.mkdir(exist_ok=True)
for p in [*P.glob('*.cuh'),*P.glob('*.inc'),P/'b1_fused.cu',P/'replace_plan.py']:(R/p.name).write_bytes(p.read_bytes())
s=(R/'b1_fused.cu').read_text().replace('#include <cuda_fp16.h>','#include <cuda_fp16.h>\n#ifndef B1_DTRI_STORE_C\n#define B1_DTRI_STORE_C 16\n#endif');(R/'b1_fused.cu').write_text(s)
p=R/'lowreg_stats.inc';p.write_text(p.read_text().replace('ch+=16)tma_store_3d','ch+=B1_DTRI_STORE_C)tma_store_3d'))
p=R/'replace_plan.py';s=p.read_text().replace('  self.xhat,self.rstd=xhat,rstd\n','  self.dtri_store_c=(defines or {}).get("B1_DTRI_STORE_C",16)\n  if self.dtri_store_c not in (16,32,64,128):raise ValueError("invalid dTri store tile")\n  self.xhat,self.rstd=xhat,rstd\n',1).replace('tm(dt,[64,16,1]','tm(dt,[64,self.dtri_store_c,1]');p.write_text(s)
