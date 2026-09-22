"""Named-barrier alternative: only the TMA issuing warp/warpgroup waits."""
from pathlib import Path
R=Path(__file__).resolve().parent
for base in ('front_prefetch_lnpair_storepipe_ctasync','front_ring112_wait256'):
 for threads in (32,128):
  s=(R/(base+'.cu')).read_text()
  helper=f'''// All warps arrive; only the loader cohort waits before the stage refill.
TMN_DEVI void release_stage(int slot){{
 if(threadIdx.x<{threads})asm volatile("bar.sync %0,256;"::"r"(slot+1):"memory");
 else asm volatile("bar.arrive %0,256;"::"r"(slot+1):"memory");
}}
'''
  pos=s.index('TMN_DEVI void input_role(');s=s[:pos]+helper+s[pos:]
  start=s.index('TMN_DEVI void input_role(');end=s.index('TMN_DEVI void reduce_at(',start)
  body=s[start:end]
  a='wgmma_wait<0>();fence_regs(acc);allsync();if(h<2)'
  assert body.count(a)==2,(base,body.count(a))
  body=body.replace(a,'wgmma_wait<0>();fence_regs(acc);release_stage(slot);if(h<2)')
  a='  if(tile+DXCOUNT<p.tiles)issue_gate('
  assert a in body
  body=body.replace(a,'  allsync(); // Protect overlapping LN/P2/P3 scratch.\n'+a)
  s=s[:start]+body+s[end:]
  name=base+f'_named_release{threads}'
  (R/(name+'.cu')).write_text(s)
  (R/(name+'.launch.json')).write_text((R/(base+'.launch.json')).read_text())
  print(name)
