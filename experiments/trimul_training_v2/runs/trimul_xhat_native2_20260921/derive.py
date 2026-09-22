from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent/'trimul_xhat_native_20260921'
for p in [*P.glob('*.cuh'),*P.glob('*.inc'),P/'b1_fused.cu',P/'replace_plan.py',P/'replace_core.py',P/'save_k3.cu',P/'bench.py']:(R/p.name).write_bytes(p.read_bytes())
s=(R/'save_k3.cu').read_text();a=s.index('    // Native tile layout:');b=s.index('    const uint32_t k0',a);block=s[a:b];s=s[:a]+s[b:]
block=block.replace('    float z[8];','    uint32_t delay=zero_dep(__float_as_uint(g0.x)^__float_as_uint(g1.x));\n    float z[8];').replace('bf16lo(fa[ks][j])','bf16lo(fa[ks][j]+delay)').replace('bf16hi(fa[ks][j])','bf16hi(fa[ks][j]+delay)')
block=block.replace('    stg128(', '    if(output)stg128(')
a=s.index('    if (CLS == math::TX) {',s.index('uint32_t chain[2]'));s=s[:a]+block+s[a:]
for j,(g,b) in enumerate([('g0','b0'),('g0','b0'),('g1','b1'),('g1','b1')]):
 ma='meanB' if j&1 else 'meanA';rs='rB' if j&1 else 'rA'
 for half,i,xy in [('lo',j*2,'x'),('hi',j*2+1,'y')]:s=s.replace('math::ln_affine(bf16%s(fa[ks][%d]), %s, %s, %s.%s, %s.%s)'%(half,j,ma,rs,g,xy,b,xy),'__fmaf_rn(z[%d],%s.%s,%s.%s)'%(i,g,xy,b,xy))
s=s.replace('lane,p.eps,tp.lnout,iw*p.N', 'lane,p.eps,(SPLITN&&cw!=0)?nullptr:tp.lnout,iw*p.N')
(R/'save_k3.cu').write_text(s)
s=(R/'replace_core.py').read_text().replace('def kernel(kind,ln=0,stats=0,pg=False,method=0):','def kernel(kind,ln=0,stats=0,pg=False,method=0,cfg=None):').replace('bi,bj,slots,acc,regs,serial=K3;', 'bi,bj,slots,acc,regs,serial=cfg or K3;').replace('smem=I.k3_smem(K3)', 'smem=I.k3_smem(cfg or K3)')
s=s.replace('def output(d,tri,ln=1,stats=0,pg=False,method=0,bufs=None):','def output(d,tri,ln=1,stats=0,pg=False,method=0,bufs=None,cfg=None):').replace('bi,bj=K3[:2];L=T._launch_module();k,smem=kernel(\'k3\',ln,stats,pg,method)', "bi,bj=(cfg or K3)[:2];L=T._launch_module();k,smem=kernel('k3',ln,stats,pg,method,cfg)")
(R/'replace_core.py').write_text(s)
s=(R/'b1_fused.cu').read_text().replace('#pragma unroll 4','#pragma unroll B1_STREAM_AFFINE_UNROLL');(R/'b1_fused.cu').write_text(s)
s=(R/'bench.py').read_text().replace("self.a=a;self.d=a['d'];self.fp32=fp32;self.audit=False", "self.a=a;self.d=a['d'];self.fp32=fp32;self.audit=False;self.fcfg=tuple(json.loads((R/('fwd-selected-L%d.json'%self.d['n'])).read_text())['config']) if (R/('fwd-selected-L%d.json'%self.d['n'])).exists() else None")
s=s.replace("method=int(self.fp32))", "method=int(self.fp32),cfg=self.fcfg)")
s=s.replace("cfg['defines']['XHAT_FP32']=int(fp32)", "cfg=json.loads((R/('bwd-selected-L%d.json'%self.d['n'])).read_text())['config'] if (R/('bwd-selected-L%d.json'%self.d['n'])).exists() else cfg;cfg['defines']['XHAT_FP32']=int(fp32)")
(R/'bench.py').write_text(s)
s=(P/'run.sbatch').read_text().replace('trimul_xhat_native_20260921','trimul_xhat_native2_20260921');(R/'run.sbatch').write_text(s)
print('Generated native2: serialized normalized stores, configurable K3')
