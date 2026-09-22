from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent/'trimul_b1_tri_opt_20260921'
for p in [*P.glob('*.cuh'),*P.glob('*.inc'),P/'b1_fused.cu',P/'replace_plan.py']:(R/p.name).write_bytes(p.read_bytes())
p=R/'b1_fused.cu';s=p.read_text().replace('#include <cuda_fp16.h>', '''#include <cuda_fp16.h>
#ifndef B1_SPLIT_DN
#define B1_SPLIT_DN 0
#endif
#define B1_STATS_BASE(slot) (B1_SPLIT_DN && (slot) ? 227840 : 225280)
''')
s=s.replace('sm+225280','sm+B1_STATS_BASE(slot)').replace('sm+225536','sm+B1_STATS_BASE(slot)+256')
# No global output nor arithmetic changes; only on-chip dNorm storage.
p.write_text(s)
p=R/'lowreg_stats.inc';s=p.read_text().replace('sm+225280','sm+B1_STATS_BASE(slot)')
s=s.replace('sn+n*8192', '(B1_SPLIT_DN ? (n<2?sm:sm+16384+slot*49152)+(n%2)*8192 : sn+n*8192)')
needle='float c1a=stats[ra*2]+stats[128+ra*2]'
s=s.replace(needle, '''#if B1_SPLIT_DN
 // All WGMMA dNorm operands are released. dNorm lives in dead dy/x_n
 // storage, so the opposite 48 KiB slot can prefetch the complete next input.
 // Stats use a distinct slot too: current mu/rstd cannot be overwritten.
 if(m0/64+UCOUNT<p.tiles)load_raw(p,sm,bars,1-slot,m0+UCOUNT*64);
#endif
 '''+needle)
s=s.replace('#if B1_EARLY_RAW', '#if B1_EARLY_RAW && !B1_SPLIT_DN');p.write_text(s)
p=R/'replace_plan.py';s=p.read_text().replace("k.set_max_dynamic_smem(227840)","smem=227840+512*int(dict(defines).get('B1_SPLIT_DN',0));k.set_max_dynamic_smem(smem)").replace("return k,u.kernel('b1_reduce'),227840","return k,u.kernel('b1_reduce'),smem")
s=s.replace("  self.xhat,self.rstd=xhat,rstd", "  if (defines or {}).get('B1_SPLIT_DN') and ((defines or {}).get('B1_PREFETCH_DY') or not (defines or {}).get('B1_DIRECT_DP')):raise ValueError('split dNorm requires direct dProj and disables dy prefetch')\n  self.xhat,self.rstd=xhat,rstd",1);p.write_text(s)
print('Generated split-dNorm prefetch candidate; GPU validation pending')
