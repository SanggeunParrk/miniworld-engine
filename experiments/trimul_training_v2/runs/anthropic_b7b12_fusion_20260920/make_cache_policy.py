from pathlib import Path
p=Path(__file__).resolve().parent;s=(p/'front_kindunroll8_inter.cu').read_text()
helper=''
for name,pol in [('last','evict_last'),('first','evict_first')]:
 helper+='''TMN_DEVI void tma_%s(void* dst,const CUtensorMap* map,uint64_t* bar,int c0,int c1){
 asm volatile("{ .reg .b64 policy; createpolicy.fractional.L2::%s.b64 policy, 1.0; cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%%0], [%%1, {%%3,%%4}], [%%2], policy; }"::"r"(smem_u32(dst)),"l"(map),"r"(smem_u32(bar)),"r"(c0),"r"(c1):"memory");
}
'''%(name,pol)
pos=s.index('TMN_DEVI void mma_weight64');s=s[:pos]+helper+s[pos:]
for mode in range(1,7):
 lines=s.splitlines()
 for i,line in enumerate(lines):
  if 'tma_load_2d(' not in line:continue
  if mode in [1,3,5,6] and ('&p.wr:' in line or '&p.wrg:' in line or '&p.wgate' in line):lines[i]=line.replace('tma_load_2d(','tma_last(')
  if mode in [2,3,4,5] and ('&p.pre' in line or '&p.dr:' in line):lines[i]=line.replace('tma_load_2d(','tma_last(')
  if mode in [4,5,6] and '&p.xn' in line:lines[i]=line.replace('tma_load_2d(','tma_last(')
  if mode in [3,4,5,6] and ('&p.x,' in line or '&p.res' in line or '&p.dg,' in line):lines[i]=line.replace('tma_load_2d(','tma_first(')
 name='front_kindcache%d'%mode;(p/(name+'.cu')).write_text('\n'.join(lines)+'\n');(p/(name+'.launch.json')).write_text((p/'front_kindunroll8_inter.launch.json').read_text())
