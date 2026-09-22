from pathlib import Path
R=Path(__file__).resolve().parent;S=R.parent/'trimul_b7_two_wg_lowreg_20260922'
for name in ('plan.py','single_wg.inc','precompile.py','probe.py','ncu_profile.py','sanitize.py','run.slurm','sanitize.slurm','profile.slurm','sweep.py'):
 (R/name).write_text((S/name).read_text().replace(S.name,R.name))
s=(S/'pair_producer.inc').read_text();i=s.index(' if(threadIdx.x)return;');h,t=s[:i],s[i:]
h=h.replace('slot=phase%2','slot=0').replace('phase>=2','phase>=1').replace('((phase/2)-1)&1','(phase-1)&1').replace('sm+g*32768+slot*16384','sm+g*16384')
h=h.replace('bar+9+g*4,1','bar+8+g*4,1')
a=t.index('  for(int stage=0;stage<16;++stage)');b=t.index('  // Both WGMMA',a)
t=t[:a]+'''  for(int batch=0;batch<8;++batch){int ws=batch%2;
   if(batch>=2)mbar_wait(bar+2+ws,((batch/2)-1)&1);
   mbar_arrive_expect_tx(bar+ws,32768);
   for(int sub=0;sub<2;++sub){int stage=batch*2+sub;tma_load_2d(sm+32768+ws*32768+sub*16384,p.wt+stage/4,bar+ws,(stage%4)*64,0);}
  }
'''+t[b:]
(R/'pair_producer.inc').write_text(h+t)
s=(S/'joint.cu').read_text();a=s.index('  for(int batch=0;batch<8;++batch)',s.index('TMN_DEVI void consumer_compute'));b=s.index('  mbar_wait(bar+19,round&1);',a)
s=s[:a]+'''  for(int batch=0;batch<8;++batch){int ws=batch%2;
   mbar_wait(bar+8+g*4,batch&1);mbar_wait(bar+ws,(batch/2)&1);
   for(int sub=0;sub<2;++sub){int stage=batch*2+sub;uint8_t* ds=sm+g*16384+sub*8192;
    static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
     input128(acc,smem_desc(smem_u32(ds+k*2048),16,1024,1),smem_desc(smem_u32(sm+32768+ws*32768+sub*16384+k*32),16,1024,1),stage>0||k>0);
    });
   }
   wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();
   if(tid==0){mbar_arrive(bar+2+ws);mbar_arrive(bar+10+g*4);}
  }
'''+s[b:];(R/'joint.cu').write_text(s)
