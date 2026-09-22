from pathlib import Path
p=Path(__file__).resolve().parent;s=(p/'front_prefetch_lnpair.cu').read_text()
for alg in ['tanh','rcpnofma']:
 if alg=='tanh':helper='TMN_DEVI float fast_gate(float x){float t;asm("tanh.approx.f32 %0,%1;":"=f"(t):"f"(x*.5f));return fmaf(t,.5f,.5f);}\n'
 else:helper='TMN_DEVI float fast_gate(float x){float e,r;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(e):"f"(__fmul_rn(x,-1.4426950408889634f)));e=__fadd_rn(1.f,e);asm("rcp.approx.ftz.f32 %0,%1;":"=f"(r):"f"(e));return r;}\n'
 a=s.index('TMN_DEVI void glu_small');v=s[:a]+helper+s[a:];v=v.replace('math::sigmoid(bf16lo(gl))','fast_gate(bf16lo(gl))').replace('math::sigmoid(bf16hi(gl))','fast_gate(bf16hi(gl))');name='front_prefetch_lnpair_'+alg;(p/(name+'.cu')).write_text(v);(p/(name+'.launch.json')).write_text((p/'front_kindprefetch.launch.json').read_text())
