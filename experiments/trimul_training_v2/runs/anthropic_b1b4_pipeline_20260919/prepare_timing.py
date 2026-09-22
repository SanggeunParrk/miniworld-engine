"""Generate instrumented copies. Never modify either original A/B kernel.

Six shared-memory timestamps avoid carrying timer registers across the body.
%globaltimer gives a common ns timebase across SMs (not a DVFS clock rate).
The instrumentation is diagnostic, and must be A/B timed against its original.
"""
from pathlib import Path
import argparse

R = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--sources', nargs='+', default=['dual', 'dual_dspref'])
parser.add_argument('--timer-bits', type=int, choices=[32, 64], default=64)
args = parser.parse_args()
for stem in args.sources:
    src = (R / f"{stem}.cu").read_text()
    # Annotated production candidate has the same code with extra comment lines.
    # Remove full-line comments only; retain code spacing used by insertion points.
    src = '\n'.join(line for line in src.splitlines()
                    if not line.lstrip().startswith('//')) + '\n'
    src = src.replace(
        'extern "C" __global__ __launch_bounds__(256,1)',
        '''// Diagnostic timestamp: only thread 0 writes each shared slot.
TMN_DEVI unsigned long long phase_ns() {
 unsigned long long t;
 asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
 return t;
}
extern "C" __global__ __launch_bounds__(256,1)''',
        1,
    )
    src = src.replace(
        '__shared__ uint64_t bar;',
        '__shared__ uint64_t bar; __shared__ unsigned long long stamps[6];\n'
        ' if(threadIdx.x==0)stamps[0]=phase_ns();',
        1,
    )
    needle = ' static_for<3>([&](auto ni){constexpr int n=decltype(ni)::value;int t=wi*3+n,tile='
    assert src.count(needle) == 1
    src = src.replace(needle, ' if(threadIdx.x==0)stamps[1]=phase_ns();\n' + needle)
    src = src.replace(
        '#if PART_ONLY == 2\n __threadfence();allsync();',
        '// Fence + CTA sync is charged to publishing the partials. All stores\n'
        ' // must be globally visible before thread 0 publishes the CTA ticket.\n'
        ' __threadfence();allsync();\n'
        ' if(threadIdx.x==0)stamps[2]=phase_ns();\n'
        '#if PART_ONLY == 2',
        1,
    )
    src = src.replace(
        ' allsync();\n for(int i=blockIdx.x*256+threadIdx.x;i<49664;',
        ' allsync();\n if(threadIdx.x==0)stamps[3]=phase_ns();\n'
        ' for(int i=blockIdx.x*256+threadIdx.x;i<49664;',
        1,
    )
    src = src.replace(
        ' if(threadIdx.x==0&&atomicAdd(p.counts+1,1u)==UCOUNT-1)',
        ' if(threadIdx.x==0)stamps[4]=phase_ns();\n'
        ' if(threadIdx.x==0&&atomicAdd(p.counts+1,1u)==UCOUNT-1)',
        1,
    )
    needle = '#endif\n\n}\nextern "C" __global__ void unified_reduce'
    assert src.count(needle) == 1
    src = src.replace(needle, '''#else
 if(threadIdx.x==0){stamps[3]=stamps[2];stamps[4]=stamps[2];}
#endif
 // Diagnostic stores only; this unused workspace is resized by the harness.
 if(threadIdx.x==0){
  stamps[5]=phase_ns();
  auto* out=reinterpret_cast<unsigned long long*>(p.groupln)+split*6;
  for(int i=0;i<6;++i)out[i]=stamps[i];
 }
}
extern "C" __global__ void unified_reduce''')
    comments = {
        'TMN_DEVI void allsync()': '// Barrier 0: all 256 CTA threads, convergent at every call.\n',
        ' if(threadIdx.x<64){mus': '// Publish saved row statistics and gamma before either WG reads them.\n',
        '  float acc[32]={};': '// WGMMA fence orders initialized accumulators; compiler fences prevent motion.\n',
        '  static_for<8>': '// Commit this dnorm chain and wait before BF16 packing or operand reuse.\n',
        '  sync_group();': '// WG barrier completes all stmatrix stores for this dnorm channel block.\n',
        ' if(lane%4==0){stats': '// CTA barrier publishes both WGs\' channel partials for each LN row.\n',
        ' if(next&&threadIdx.x==0)': '// Next parity: expect 128 KiB; issue 80 KiB now, remaining 48 KiB in dual_load.\n',
        ' sync_group();fence_proxy_async();': '// Complete generic dtri writes, proxy-fence, and reconverge WG before TMA reads.\n'
            '// Commit/wait all TMA stores before this WG allows shared dtri reuse.\n',
        ' if(threadIdx.x==0){if(first)': '// First parity expects 128 KiB rows + 64 KiB resident Wp, with one issuer.\n',
        ' allsync();mbar_wait': '// CTA joins descriptor issue; parity wait acquires every byte before B1 reads.\n',
        ' fence_proxy_async();allsync();': '// Publish generic dp/dg shared stores to the WGMMA async proxy, then join CTA.\n',
        ' if(threadIdx.x==0){mbar_init': '// Initialize one-arrival mbarrier and publish initialization before CTA sync.\n',
        ' for(int i=threadIdx.x;i<512;': '// Join after LN partial initialization so either WG can safely update its channels.\n',
        ' for(int it=first;it<end;': '// Flip parity once per owned row tile; an empty CTA skips all TMA waits.\n',
        '  static_for<3>': '// Fence each retained dW fragment before WGMMA; commit three independent chains.\n',
        '   static_for<4>': '// Wait for all committed dW chains, then compiler-fence before fragment reuse.\n',
        '  dual_dgrad(p,sm,': '// Join both WGs after LN and completed TMA stores before loading the next tile.\n',
        ' if(threadIdx.x==0){atomicAdd(p.counts,': '// Cooperative grid guarantees residency; all partials were fenced before ticket.\n',
        ' allsync();\n if(threadIdx.x==0)stamps[3]': '// Join thread 0 after every CTA has published; volatile loads avoid stale L1.\n',
        ' __threadfence();allsync();\n if(threadIdx.x==0)stamps[4]': '// Publish final outputs and join all CTA threads before the completion ticket.\n',
    }
    for needle, comment in comments.items():
        assert needle in src, (stem, needle)
        src = src.replace(needle, comment + needle, 1)
    if args.timer_bits == 32:
        # Lower live-register cost for the register-saturated optimized kernel.
        # read_phases rejects a launch crossing the 4.29-second timer wrap.
        src = src.replace('TMN_DEVI unsigned long long phase_ns()',
                          'TMN_DEVI unsigned int phase_ns()')
        src = src.replace(' unsigned long long t;', ' unsigned int t;')
        src = src.replace('mov.u64 %0, %%globaltimer;',
                          'mov.u32 %0, %%globaltimer_lo;')
        src = src.replace('"=l"(t)', '"=r"(t)')
        src = src.replace('__shared__ unsigned long long stamps[6];',
                          '__shared__ unsigned int stamps[6];')
    out = R / f"{stem}_timing.cu"
    out.write_text('// Generated by prepare_timing.py; diagnostic only.\n' + src)
    print(out.name)
