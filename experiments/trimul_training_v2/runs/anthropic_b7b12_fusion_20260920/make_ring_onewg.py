from pathlib import Path
import re
p=Path(__file__).resolve().parent;s=(p/'front_ring96_cache3.cu').read_text();old=(p/'front_onewg_lowreg.cu').read_text()
s=s.replace('sm+16384+n*16384+k*8192','sm+16384+k*16384+n*8192')
a=s.index('TMN_DEVI void input_role');b=s.index('TMN_DEVI void reduce_at',a);v=s[a:b]
v=v.replace(' int split=',' __shared__ volatile int done;int tid=threadIdx.x;if(tid==0)done=0;allsync();if(tid>=128){setmaxnreg_dec<32>();while(!done)__nanosleep(32);setmaxnreg_inc<128>();return;}setmaxnreg_inc<192>();\n int split=',1)
v=v.replace('float running=0;','float running_g=0,running_b=0;').replace('gate_packed[16];float acc[32]','gate_packed[32];float acc[64]').replace('float gate[32]','float gate[64]').replace('mma_dgrad(gate','mma_gate128(gate').replace('sm+16384+wi*16384+(k/4)*8192','sm+16384+(k/4)*16384')
v=v.replace('static_for<16>','static_for<32>').replace('mma_input64(acc','mma_front128(acc').replace('s+24576+wi*8192','s+24576').replace('s+wi*8192+k*32','s+k*32')
start=v.index('  // B10 outputs BF16');end=v.rfind('\n}')
oa=old.index('  static_for<32>([&](auto qi){constexpr int q=decltype(qi)::value;acc[q*2]');ob=old.index('TMN_DEVI void reduce_at',oa);ln=old[oa:ob]
la=ln.index('  if(tid==0){mbar_arrive_expect_tx');lb=ln.index('  float mu[2]',la);ln=ln[:la]+'  uint8_t* lnsm=sm+65536;mbar_wait(bar+3,round&1);\n'+re.sub(r'\bsm\b','lnsm',ln[lb:]);ln=ln.replace('packed[q]','gate_packed[q]')
ln=ln[:ln.rfind('\n}')]+ '\n setmaxnreg_dec<128>();if(tid==0)done=1;\n}\n'
v=v[:start]+ln
# This CTA role has a single active WG; role-level rendezvous above stays256.
start=v.index('setmaxnreg_inc<192>();')+len('setmaxnreg_inc<192>();');v=v[:start]+v[start:].replace('allsync();','named_bar_sync(1,128);')
# ring_ready's256-thread rendezvous must likewise be local to the active WG.
v=v.replace('ring_ready(p,tile);','if(tid<8)ring_wait(p.counts+2+(tile%RING_TILES)*8+tid,tile+1);named_bar_sync(1,128);')
s=s[:a]+v+s[b:];name='front_ring96_onewg';(p/(name+'.cu')).write_text(s);(p/(name+'.launch.json')).write_text((p/'front_ring96_cache3.launch.json').read_text())
