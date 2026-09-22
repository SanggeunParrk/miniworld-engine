"""Bound independent-WG register lifetime by reloading raw tri from shared."""
from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'dual_wg_tiles.cu').read_text()
s=s.replace('uint32_t fx[4][4][4],dn[4][4][4];','uint32_t dn[4][4][4];')
s=s.replace('   ldsm_x4_t(fx[nlocal][q],', '   uint32_t fx[4];\n   ldsm_x4_t(fx,')
s=s.replace('fx[nlocal][q][j]','fx[j]')
s=s.replace('uint32_t out[4];', '''uint32_t out[4],fx[4];
   // Each warp reloads only the channel/row block it will overwrite; all
   // values are consumed before that warp's matching stmatrix instruction.
   ldsm_x4_t(fx,smem_u32(sx)+swz128(n*64+q*16+r8+8*(mat>>1),(w*16+8*(mat&1))*2));''')
s=s.replace('fx[nl][q][j+1]', 'fx[j+1]').replace('fx[nl][q][j]', 'fx[j]')
s=s.replace('s1[2]={},s2[2]={},v1', 'v1')
(r/'dual_wg_reload.cu').write_text('// Experiment: independent WG with bounded raw-tri lifetime.\n'+s)
(r/'dual_wg_reload.py').write_text('from dual_experiment import Experiment\nclass Plan(Experiment):\n    def __init__(self,d,dy,saved,count=132,part=2):\n        super().__init__(d,dy,saved,count,part,"dual_wg_reload")\n')
