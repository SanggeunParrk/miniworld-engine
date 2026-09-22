"""Preserve release/acquire synchronization while avoiding acquire on failed polls.

PTX memory model section8.8: a strong relaxed read followed by an acquire fence
forms an acquire pattern. Use acq_rel.gpu, supported by the CUDA12.9 toolchain.
The producer's release and its preceding full TMA-store completion are unchanged.
"""
from pathlib import Path
R=Path(__file__).resolve().parent
for base,delay in [('front_ring96_cache3',32),('front_ring112_wait256',256)]:
 for fast in (False,True):
  s=(R/(base+'.cu')).read_text();a=s.index('TMN_DEVI void ring_wait(');b=s.index('TMN_DEVI void ring_publish',a)
  fastpath='''asm volatile("ld.acquire.gpu.global.u32 %0,[%1];":"=r"(got):"l"(ptr):"memory");if(got>=want)return;''' if fast else ''
  fn='''TMN_DEVI void ring_wait(const unsigned* ptr,unsigned want){unsigned got;
'''+fastpath+'''
 do{asm volatile("ld.relaxed.gpu.global.u32 %0,[%1];":"=r"(got):"l"(ptr):"memory");
  if(got<want)__nanosleep('''+str(delay)+''');
 }while(got<want);
 asm volatile("fence.acq_rel.gpu;":::"memory");
}
'''
  name=base+('_poll_fast' if fast else '_poll_fence')
  (R/(name+'.cu')).write_text(s[:a]+fn+s[b:])
  (R/(name+'.launch.json')).write_text((R/(base+'.launch.json')).read_text())
  print(name)
