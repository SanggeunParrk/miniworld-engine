from pathlib import Path
p=Path(__file__).resolve().parent
for base in ['front_prefetch_lnpair_storepipe','front_ring96_cache3_storepipe','front_ring128_cache3_storepipe']:
 for sleep in [32,128]:
  s=(p/(base+'.cu')).read_text();a=s.index('TMN_DEVI void reduce_at');helper='''TMN_DEVI void wait_total(const unsigned* ptr){unsigned got;do{asm volatile("ld.acquire.gpu.global.u32 %%0,[%%1];":"=r"(got):"l"(ptr):"memory");if(got!=UCOUNT)__nanosleep(%d);}while(got!=UCOUNT);}\n'''%sleep;s=s[:a]+helper+s[a:]
  old='while(atomicAdd(p.counts,0u)!=UCOUNT)__nanosleep(32);';assert old in s;s=s.replace(old,'wait_total(p.counts);');name=base+'_acq%d'%sleep;(p/(name+'.cu')).write_text(s);(p/(name+'.launch.json')).write_text((p/(base+'.launch.json')).read_text())
