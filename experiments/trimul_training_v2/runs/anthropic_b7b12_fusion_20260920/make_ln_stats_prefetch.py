from pathlib import Path
import json
p=Path(__file__).resolve().parent
helper='''TMN_DEVI void load_stats(float* dst,const float* src,uint64_t* bar){asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],256,[%2];"::"r"(smem_u32(dst)),"l"(src),"r"(smem_u32(bar)):"memory");}
'''
for base in ['front_prefetch_lnpair_storepipe','front_ring96_cache3','front_ring96_queue_wtma128']:
 s=(p/(base+'.cu')).read_text();pos=s.index('TMN_DEVI void mma_weight64');s=s[:pos]+helper+s[pos:]
 a=s.index('TMN_DEVI void load_ln_next');b=s.index('\nTMN_DEVI',a+1);f=s[a:b]
 # After right P1, G3 has been consumed and P3 weights end at57344.
 # Next gate occupies only0..49152. The512-byte gap57344..57856 is
 # therefore free until next tile G loads, after the current LN completes.
 f=f.replace('mbar_arrive_expect_tx(bar,32768);','mbar_arrive_expect_tx(bar,33280);load_stats(reinterpret_cast<float*>(sm-8192),p.mean+row,bar);load_stats(reinterpret_cast<float*>(sm-7936),p.rs+row,bar);')
 s=s[:a]+f+s[b:]
 for old,new in [('p.mean[row+ra]','reinterpret_cast<float*>(sm+57344)[ra]'),('p.mean[row+rb]','reinterpret_cast<float*>(sm+57344)[rb]'),('p.rs[row+ra]','reinterpret_cast<float*>(sm+57600)[ra]'),('p.rs[row+rb]','reinterpret_cast<float*>(sm+57600)[rb]')]:
  assert old in s;s=s.replace(old,new)
 name=base+'_lnstats_gap';(p/(name+'.cu')).write_text(s)
 cfg=json.loads((p/(base+'.launch.json')).read_text())
 (p/(name+'.launch.json')).write_text(json.dumps(cfg))
