"""Overlap multicast x_n arrival with the independent B7 GLU computation.

Preactivation/upstream and x_n use separate transaction barriers. B7 consumes
only the former; WGMMA waits for x_n after B7 and the ring-store launch. This
preserves workspace, contractions and every BF16 rounding point.
"""
from pathlib import Path
R=Path(__file__).resolve().parent
for size in (4,8):
 base='front_ring96_cache3_mcastasync_xn'+str(size)
 s=(R/(base+'.cu')).read_text()
 a=s.index('TMN_DEVI void load_dw(');b=s.index('\n}',a)+2
 f=s[a:b]
 f=f.replace('mbar_arrive_expect_tx(b+slot,40960);pair_ready(b+slot+4,(row/(64*DW_SPLITS)/2)&1);',
             'mbar_arrive_expect_tx(b+slot,24576);mbar_arrive_expect_tx(b+slot+8,16384);')
 old='for(int n=0;n<2;++n)tma_pair(s+24576+n*8192,&p.xn,b+slot,n*64,row);'
 assert old in f
 f=f.replace(old,'pair_ready(b+slot+4,(row/(64*DW_SPLITS)/2)&1);\n '+old.replace('b+slot,n*64','b+slot+8,n*64'))
 s=s[:a]+f+s[b:]
 old='ring_begin(p,s,tile,group);\n  fence_regs(acc);'
 assert old in s
 s=s.replace(old,'ring_begin(p,s,tile,group);mbar_wait(b+slot+8,(r/2)&1);\n  fence_regs(acc);')
 s=s.replace('__shared__ uint64_t bar[8]','__shared__ uint64_t bar[10]')
 s=s.replace('fence_barrier_init();}allsync();cluster_sync_pair();',
             'for(int i=8;i<10;++i)mbar_init(bar+i,1);fence_barrier_init();}allsync();cluster_sync_pair();')
 name=base+'_overlap'
 (R/(name+'.cu')).write_text(s)
 (R/(name+'.launch.json')).write_text((R/(base+'.launch.json')).read_text())
 print(name)
