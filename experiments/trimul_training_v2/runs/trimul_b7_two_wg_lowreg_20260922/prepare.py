from pathlib import Path
R=Path(__file__).resolve().parent;S=R.parent/'trimul_b7_two_wg_uniform_20260922'
for name in ('plan.py','single_wg.inc','pair_producer.inc','precompile.py','probe.py','ncu_profile.py','sanitize.py','run.slurm','sanitize.slurm','profile.slurm','sweep.py'):
 (R/name).write_text((S/name).read_text().replace(S.name,R.name))
s=(S/'joint.cu').read_text()
a=s.index('TMN_DEVI void consumer_compute(');b=s.index('extern "C" __global__',a);t=s[a:b]
x=t.index('  static_for<2>([&](auto hh)');y=t.index('  #define B7_ACC',x)
t=t[:x]+'''  static_for<4>([&](auto hh){constexpr int h=decltype(hh)::value;float gate[16]={};fence_regs(gate);wgmma_fence();
   static_for<8>([&](auto kk){constexpr int k=decltype(kk)::value;
    gate32(gate,smem_desc(smem_u32(sm+g*32768+(k/4)*8192+(k%4)*32),16,1024,1),smem_desc(smem_u32(sm+65536+(k/4)*16384+h*4096+(k%4)*32),16,1024,1),k>0);
   });wgmma_commit();wgmma_wait<0>();fence_regs(gate);
   static_for<8>([&](auto jj){constexpr int j=decltype(jj)::value;
    packed_dx[h*8+j]=pack_bf16(acc[h*16+j*2]+math::round_bf16(gate[j*2]),acc[h*16+j*2+1]+math::round_bf16(gate[j*2+1]));
   });
  });
  }
'''+t[y:]
old='''  uint32_t raw[8][4];load_frag_bf16<8,8192>(raw,smem_u32(lnsm),w*16,lane);
  LnStats st=ln_stats_only<8>(raw,gamma,beta,lane,1e-5f);'''
assert old in t
t=t.replace(old,'''  LnStats st;
  {uint32_t raw[8][4];load_frag_bf16<8,8192>(raw,smem_u32(lnsm),w*16,lane);st=ln_stats_only<8>(raw,gamma,beta,lane,1e-5f);}''')
start=t.index('  float mu[2]')
head,tail=t[:start],t[start:]
old='static_for<8>([&](auto qq){constexpr int q=decltype(qq)::value;'
new=old+'''
   uint32_t rawq[4];ldsm_x4(rawq,smem_u32(lnsm)+(q/4)*8192+swz128(w*16+(lane&7)+((lane&8)?8:0),((q%4)*16+((lane&16)?8:0))*2));'''
assert tail.count(old)==2
tail=tail.replace(old,new).replace('raw[q][','rawq[')
# All rows' raw data must remain intact until the last streamed reread.
# Store output in the dead dGate/residual buffer only after reading residual.
tail=tail.replace('uint8_t* out=lnsm+(q/4)*8192;', 'uint8_t* out=lnsm-16384+(q/4)*8192;')
tail=tail.replace('store2d(&p.dx,lnsm,0,row);store2d(&p.dx,lnsm+8192,64,row);','store2d(&p.dx,lnsm-16384,0,row);store2d(&p.dx,lnsm-8192,64,row);')
t=head+tail;s=s[:a]+t+s[b:];(R/'joint.cu').write_text(s)
