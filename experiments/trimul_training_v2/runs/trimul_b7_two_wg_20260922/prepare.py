from pathlib import Path
R=Path(__file__).resolve().parent
src=R.parent/'trimul_b7_384_n64_20260922'
for name in ('single_wg.inc','precompile.py','probe.py','ncu_profile.py','sanitize.py','run.slurm','sanitize.slurm','profile.slurm','sweep.py'):
 (R/name).write_text((src/name).read_text().replace(src.name,R.name))
p=(src/'plan.py').read_text().replace('self.count=clusters*(16+self.consumers)','self.count=clusters*(16+self.consumers//2)')
p=p.replace("(R/'joint.cu',R/'single_wg.inc')", "(R/'joint.cu',R/'single_wg.inc',R/'pair_producer.inc')")
p=p.replace('(16+self.consumers)%self.hwcluster==0','self.count%self.hwcluster==0 and self.consumers%2==0 and self.multicast==0')
(R/'plan.py').write_text(p)
s=(src/'joint.cu').read_text().replace('GROUP=SOURCES+CONSUMERS','GROUP=SOURCES+CONSUMERS/2')
a=s.index('TMN_DEVI void consumer_producer(');b=s.index('TMN_DEVI void wide_gate(',a)
s=s[:a]+'#include "pair_producer.inc"\n'+s[b:]
a=s.index('TMN_DEVI void consumer_compute(');b=s.index('extern "C" __global__',a)
t=s[a:b]
header='''TMN_DEVI void consumer_compute(const Params& p,uint8_t* sm,uint64_t* bar,const float* gamma,const float* beta){
 int g=threadIdx.x/128-1,rank=2*(blockIdx.x%GROUP-SOURCES)+g,cid=blockIdx.x/GROUP,clusters=gridDim.x/GROUP,round=0,tid=threadIdx.x%128,lane=tid%32,w=tid/32,ra=w*16+lane/4,rb=ra+8;float run_g=0,run_b=0;
 for(int base=cid+(rank-g)*clusters;base<p.tiles;base+=CONSUMERS*clusters,++round){int tile=base+g*clusters,row=tile*64;bool valid=tile<p.tiles;
  uint32_t packed_dx[32];
  {
  float acc[64]={};fence_regs(acc);wgmma_fence();
  for(int batch=0;batch<8;++batch){
   for(int sub=0;sub<2;++sub){int stage=batch*2+sub,slot=batch%2,ws=stage%2;
    if(sub==0&&valid)mbar_wait(bar+8+g*4+slot,(batch/2)&1);
    mbar_wait(bar+ws,(stage/2)&1);
    if(valid){uint8_t* ds=sm+g*32768+slot*16384+sub*8192;
     static_for<4>([&](auto kk){constexpr int k=decltype(kk)::value;
      input128(acc,smem_desc(smem_u32(ds+k*2048),16,1024,1),smem_desc(smem_u32(sm+65536+ws*16384+k*32),16,1024,1),stage>0||k>0);
     });wgmma_commit();
    }
   }
   wgmma_wait<0>();fence_regs(acc);allsync();
   if(tid==0){for(int ws=0;ws<2;++ws)mbar_arrive(bar+2+ws);if(valid)mbar_arrive(bar+10+g*4+batch%2);}
  }
  mbar_wait(bar+19,round&1);
  static_for<2>([&](auto hh){constexpr int h=decltype(hh)::value;float gate[32]={};fence_regs(gate);wgmma_fence();
   static_for<8>([&](auto kk){constexpr int k=decltype(kk)::value;
    gp_ss64(gate,smem_desc(smem_u32(sm+g*32768+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sm+65536+(k/4)*16384+h*8192+(k%4)*32),16,1024,1),k>0);
   });wgmma_commit();wgmma_wait<0>();fence_regs(gate);
   static_for<32>([&](auto jj){constexpr int j=decltype(jj)::value;acc[h*32+j]=math::round_bf16(acc[h*32+j]+math::round_bf16(gate[j]));});
  });
  static_for<32>([&](auto jj){constexpr int j=decltype(jj)::value;packed_dx[j]=pack_bf16(acc[j*2],acc[j*2+1]);});
  }
  #define B7_ACC(i) (((i)&1)?bf16hi(packed_dx[(i)/2]):bf16lo(packed_dx[(i)/2]))
  allsync();
  // dGate is dead after WGMMA; reuse its 16 KiB for residual, not another buffer.
  if(tid==0){mbar_arrive_expect_tx(bar+20+g,16384);for(int k=0;k<2;++k)tma_load_2d(sm+g*32768+k*8192,&p.res,bar+20+g,k*64,row);}
  uint8_t* lnsm=sm+g*32768+16384;
'''
ln=t[t.index('  uint32_t raw[8][4];'):]
ln=ln.replace('float* tmp=reinterpret_cast<float*>(lnsm+32768);','float* tmp=reinterpret_cast<float*>(sm+98304+g*4096);\n  mbar_wait(bar+20+g,round&1);')
ln=ln.replace('uint8_t* res=lnsm+16384+(q/4)*8192;','uint8_t* res=lnsm-16384+(q/4)*8192;')
ln=ln.replace('if(tid==0){store2d(', 'if(tid==0&&valid){store2d(')
ln=ln.replace('if(tid==0)tma_store_wait_all();','if(tid==0&&valid)tma_store_wait_all();')
ln=ln.replace('mbar_arrive(bar+17)','mbar_arrive(bar+18)')
s=s[:a]+header+ln+s[b:]
s=s.replace('__shared__ uint64_t bar[40]','__shared__ uint64_t bar[64]')
s=s.replace('for(int i=0;i<40;++i)mbar_init(bar+i,(B7_MULTICAST&&i>=32&&i<36&&rank<SOURCES)?B7_HW_CLUSTER:1);','for(int i=0;i<64;++i)mbar_init(bar+i,(rank>=SOURCES&&(i==2||i==3||i==18))?2:1);')
s=s.replace('if(wi==2){setmaxnreg_dec<24>();}\n else if(rank<SOURCES){','if(rank<SOURCES){\n  if(wi==2){setmaxnreg_dec<24>();}else{')
old=''' }else{
  if(wi==0){setmaxnreg_dec<B7_PRODUCER_REGS>();consumer_producer(p,sm,bar);}else{setmaxnreg_inc<216-B7_PRODUCER_REGS>();consumer_compute(p,sm,bar,gamma,beta);}
 }'''
new=''' }}else{
  if(wi==0){setmaxnreg_dec<32>();pair_producer(p,sm,bar);}else{setmaxnreg_inc<104>();consumer_compute(p,sm,bar,gamma,beta);}
 }'''
assert old in s
s=s.replace(old,new)
(R/'joint.cu').write_text(s)
print('Prepared two-WG source and plan')
