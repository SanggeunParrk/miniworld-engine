from pathlib import Path
R=Path(__file__).resolve().parent
src=R.parent/'trimul_b7_ring_cluster_schedule_20260922'
for name in ('single_wg.inc','precompile.py','probe.py','ncu_profile.py','sanitize.py','run.slurm','sanitize.slurm','profile.slurm'):
    (R/name).write_text((src/name).read_text().replace(src.name,R.name))
p=(src/'plan.py').read_text().replace("flags=['-DB7_HW_CLUSTER=", "flags=['-DB7_PACK_DX='+str(int(os.environ.get('B7_PACK','0'))&1),'-DB7_PACK_GP='+str((int(os.environ.get('B7_PACK','0'))>>1)&1),'-DB7_HW_CLUSTER=")
(R/'plan.py').write_text(p)
s=(src/'joint.cu').read_text()
start=s.index('TMN_DEVI void pair_glu(')
end=s.index('// Compact ring planes:',start)
packed=s[start:end].replace('pair_glu(float (&a)[32]','pair_glu_packed(uint32_t (&a)[16]')
packed=packed.replace('pack_bf16(a[q*8+j*2],a[q*8+j*2+1])','a[q*4+j]').replace('pack_bf16(a[(q+2)*8+j*2],a[(q+2)*8+j*2+1])','a[(q+2)*4+j]')
s=s[:end]+packed+s[end:]
old='float a0[32]={},a1[32]={};pair_gp(a0,xn,desc);wgmma_wait<0>();fence_regs(a0);'
assert s.count(old)==1
s=s.replace(old,'''float a1[32]={};
#if B7_PACK_GP
  uint32_t packed_a0[16];
  {float a0[32]={};pair_gp(a0,xn,desc);wgmma_wait<0>();fence_regs(a0);
   static_for<16>([&](auto jj){constexpr int j=decltype(jj)::value;packed_a0[j]=pack_bf16(a0[j*2],a0[j*2+1]);});}
#else
  float a0[32]={};pair_gp(a0,xn,desc);wgmma_wait<0>();fence_regs(a0);
#endif''')
s=s.replace('  pair_glu(a0,s0,sg0,ma,mb);','''#if B7_PACK_GP
  pair_glu_packed(packed_a0,s0,sg0,ma,mb);
#else
  pair_glu(a0,s0,sg0,ma,mb);
#endif''')
start=s.index('TMN_DEVI void consumer_compute(')
end=s.index('extern "C" __global__',start)
t=s[start:end]
t=t.replace('  float acc[64]={};','''#if B7_PACK_DX
  uint32_t packed_dx[32];
  {
#endif
  float acc[64]={};''')
old='  uint8_t* lnsm=sm+LN;'
t=t.replace(old,'''#if B7_PACK_DX
  static_for<32>([&](auto jj){constexpr int j=decltype(jj)::value;packed_dx[j]=pack_bf16(acc[j*2],acc[j*2+1]);});
  }
  #define B7_ACC(i) (((i)&1)?bf16hi(packed_dx[(i)/2]):bf16lo(packed_dx[(i)/2]))
#else
  #define B7_ACC(i) acc[i]
#endif
'''+old)
head,tail=t.split(old,1)
import re
tail=re.sub(r'acc\[([^\]]+)\]',r'B7_ACC(\1)',tail)
t=head+old+tail+'\n#undef B7_ACC\n'
s=s[:start]+t+s[end:]
(R/'joint.cu').write_text(s)

s=(src/'sweep.py').read_text()
s=s.replace("p.add_argument('--configs',default='1:10:10:0,2:10:10:0')", "p.add_argument('--configs',default='0:64,1:64,2:64,3:64');p.add_argument('--rounds',type=int,default=3);p.add_argument('--iterations',type=int,default=80)")
s=s.replace("old=module('selected_348',R.parent/'trimul_b7_ring_depth_20260922/plan.py')", "os.environ['B7_HW_CLUSTER']='2';os.environ['B7_MULTICAST']='0'\n old=module('current_selected',R.parent/'trimul_b7_ring_cluster_schedule_20260922/plan.py')")
start=s.index(' for spec in args.configs.split')
end=s.index(' limits=',start)
s=s[:start]+''' for spec in args.configs.split(','):
  variant,regs=map(int,spec.split(':'));os.environ['B7_PACK']=str(variant);os.environ['B7_PRODUCER_REGS']=str(regs)
  q=plan.Plan(d,dy,dl,dr,dg,xn=refplan.xn,clusters=10,mode=52)
  q.mask=refplan.mask;q.bind(dl,dr,dg,dy,xn=refplan.xn);plans[f'v{variant}p{regs}']=q
'''+s[end:]
s=s.replace('iterations=160) for _ in range(5)','iterations=args.iterations) for _ in range(args.rounds)')
(R/'sweep.py').write_text(s)
print('Prepared',R)
