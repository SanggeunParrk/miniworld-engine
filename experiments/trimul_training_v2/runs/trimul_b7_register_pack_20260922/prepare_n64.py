from pathlib import Path
R=Path(__file__).resolve().parent
for parent,name in [('trimul_b7_384_control_20260922','trimul_b7_384_n64_20260922'),('trimul_b7_gate_half_20260922','trimul_b7_256_n64_20260922')]:
 src=R.parent/parent;out=R.parent/name;out.mkdir(exist_ok=True)
 for f in ('plan.py','single_wg.inc','precompile.py','probe.py','ncu_profile.py','sanitize.py','run.slurm','sanitize.slurm','profile.slurm','sweep.py'):
  (out/f).write_text((src/f).read_text().replace(parent,name))
 s=(src/'joint.cu').read_text()
 helper='''TMN_DEVI void small_dw(float (&d)[64],uint64_t a,uint64_t b,int accumulate){
 static_for<2>([&](auto hh){constexpr int h=decltype(hh)::value;auto& part=*reinterpret_cast<float(*)[32]>(d+h*32);mma_recompute(part,a,b+h*(8192>>4),accumulate);});
}
TMN_DEVI void input64(float (&d)[32],uint64_t a,uint64_t b,int accumulate){
'''
 sg=(src/'single_wg.inc').read_text();asm=sg[sg.index(' asm volatile('):sg.index('// SPDX')].strip()
 asm=asm[:asm.rfind('}')].replace('p, 1, 1, 0, 0;', 'p, 1, 1, 1, 0;')
 helper+=asm+'\n}\n'
 s=s.replace('TMN_DEVI void source_compute(',helper+'\nTMN_DEVI void source_compute(')
 s=s.replace('mma_weight128(dw,','small_dw(dw,')
 start=s.index('TMN_DEVI void input128(');end=s.index('TMN_DEVI void consumer_producer',start)
 s=s[:start]+'''TMN_DEVI void input128(float (&d)[64],uint64_t a,uint64_t b,int accumulate){
 static_for<2>([&](auto hh){constexpr int h=decltype(hh)::value;auto& part=*reinterpret_cast<float(*)[32]>(d+h*32);input64(part,a,b+h*(8192>>4),accumulate);});
}
'''+s[end:]
 (out/'joint.cu').write_text(s)
 print(out)
