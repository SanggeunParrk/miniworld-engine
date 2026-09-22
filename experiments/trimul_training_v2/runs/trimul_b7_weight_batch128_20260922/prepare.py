from pathlib import Path
R=Path(__file__).resolve().parent;S=R.parent/'trimul_b7_ring_cluster_schedule_20260922'
for name in ('plan.py','single_wg.inc','precompile.py','probe.py','ncu_profile.py','sanitize.py','run.slurm','sanitize.slurm','profile.slurm'):
 (R/name).write_text((S/name).read_text().replace(S.name,R.name))
# Reuse mutation/timing harness and current selected baseline.
harness=(R.parent/'trimul_b7_register_pack_20260922/sweep.py').read_text()
harness=harness.replace("os.environ['B7_PRODUCER_REGS']=str(regs)","os.environ['B7_PRODUCER_REGS']=str(regs);os.environ['B7_RING_DEPTH']=str(variant if variant else 12)")
(R/'sweep.py').write_text(harness)
s=(S/'joint.cu').read_text().replace('RING_SLOTS=65536/RING_CHUNK','RING_SLOTS=32768/RING_CHUNK')
a=s.index('TMN_DEVI void consumer_producer(');b=s.index('TMN_DEVI void wide_gate',a);t=s[a:b]
t=t.replace('mbar_wait(bar+32+slot,0)','mbar_wait(bar+32+slot,((phase/RING_SLOTS)-1)&1)')
a2=t.index(' for(int tile=cid+rank*groups;',t.index(' if(threadIdx.x)return;'))
t=t[:a2]+''' for(int tile=cid+rank*groups;tile<p.tiles;tile+=CONSUMERS*groups,++round){int row=tile*64;
  mbar_arrive_expect_tx(bar+19,16384);for(int k=0;k<2;++k)tma_load_2d(sm+98304+k*8192,&p.dg,bar+19,k*64,row);
  for(int batch=0;batch<8;++batch){int ws=batch%2;
   if(batch>=2)mbar_wait(bar+6+ws,((batch/2)-1)&1);
   mbar_arrive_expect_tx(bar+2+ws,32768);
   for(int sub=0;sub<2;++sub){int stage=batch*2+sub;tma_load_2d(sm+32768+ws*32768+sub*16384,p.wt+stage/4,bar+2+ws,(stage%4)*64,0);}
  }
  for(int ws=0;ws<2;++ws)mbar_wait(bar+6+ws,1);
  mbar_arrive_expect_tx(bar+12,65536);
  for(int k=0;k<2;++k){tma_load_2d(sm+k*8192,&p.x,bar+12,k*64,row);tma_load_2d(sm+16384+k*8192,&p.res,bar+12,k*64,row);tma_load_2d(sm+WGATE+k*16384,&p.wgate,bar+12,k*64,0);}
  mbar_wait(bar+17,round&1);
 }
}
'''
s=s[:a]+t+s[b:]
a=s.index('  for(int batch=0;batch<8;++batch)',s.index('TMN_DEVI void consumer_compute'))
b=s.index('}mbar_wait(bar+12,round&1);',a)+1
s=s[:a]+'''  for(int batch=0;batch<8;++batch){int slot=batch%2,ws=batch%2;
   mbar_wait(bar+24+slot,(batch/2)&1);mbar_wait(bar+2+ws,(batch/2)&1);
   for(int sub=0;sub<2;++sub){int stage=batch*2+sub;uint8_t* ds=sm+slot*16384+sub*8192;
    static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
     input128(acc,smem_desc(smem_u32(ds+k*2048),16,1024,1),smem_desc(smem_u32(sm+32768+ws*32768+sub*16384+k*32),16,1024,1),stage>0||k>0);
    });
   }
   wgmma_commit();wgmma_wait<0>();fence_regs(acc);allsync();
   if(tid==0){mbar_arrive(bar+6+ws);mbar_arrive(bar+32+slot);}
  }'''+s[b:]
(R/'joint.cu').write_text(s)
