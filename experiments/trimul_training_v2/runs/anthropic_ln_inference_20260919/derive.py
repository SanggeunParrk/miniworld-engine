from pathlib import Path
R=Path(__file__).resolve().parent
E=R.parent/'trimul_sm90_parity_20260917/engine'
U=E/'third_party/anthropic/upstream/common/opt_core/opt_core/kernels/trimul/native/pkg/v5/csrc'
s=(U/'tmn_kernels.cuh').read_text()
a=s.index('template <class G, bool HAS_MASK, int LNM, bool SAVE, bool EMITX = false>\nTMN_DEVI void k1_body');b=s.index('// ============================================================================================================ K3',a)
k=s[a:b].replace('void k1_body','void infer_k1_body')
k=k.replace('for (int i = tid; i < CZ; i += G::NTHR)', 'if (MW_FUSED) for (int i = tid; i < CZ; i += G::NTHR)')
head='''// SPDX-License-Identifier: Apache-2.0
// Direct derivative of Anthropic f4f62fa native v5, no backward saves.
#include "tmn_kernels.cuh"
namespace tmn { namespace sm90 {
using Base=K1Cfg<128,256,false,MW_BI,MW_BJ,MW_SLOT,MW_SK,MW_SCHED>;
struct Cfg:Base {static constexpr int CONS_REGS=MW_REGS;};
'''
foot='''}}
extern "C" __global__ __launch_bounds__(tmn::sm90::Cfg::NTHR,tmn::sm90::Cfg::MINB)
void infer_k1(__grid_constant__ const tmn::K1Params p){tmn::sm90::infer_k1_body<tmn::sm90::Cfg,true,MW_FUSED,false,false>(p);}
'''
(R/'k1.cu').write_text(head+k+foot)
a=s.index('template <class G, int LNM, bool UPD = false>\nTMN_DEVI void k3_body');k=s[a:s.index('\n}  //',a)] if '\n}  //' in s[a:] else s[a:]
# Use the end of the body before the namespace closures.
end=k.rfind('\n}')
# Upstream ends with body then namespace sm90 / tmn; inspect with split marker below.
brace=k.index('{');depth=1;i=brace+1
while depth:
 if k[i]=='{':depth+=1
 elif k[i]=='}':depth-=1
 i+=1
k=k[:i].replace('void k3_body','void infer_k3_body')
k=k.replace('for (int i = tid; i < CZ; i += NTHREADS)', 'if (MW_FUSED) for (int i = tid; i < CZ; i += NTHREADS)')
k=k.replace('if (!ZF32 && !PRENORM) {','if (MW_FUSED && !ZF32 && !PRENORM) {')
k=k.replace('ln_fragment<KSP, true>(fx','ln_fragment<KSP, MW_SERIAL>(fx').replace('ln_fragment<KSG, true>(fz','ln_fragment<KSG, MW_SERIAL>(fz')
old='ldsm_x4(rz, sZ_u + (uint32_t)(b0 >> 1) * (uint32_t)CHB + swz128((uint32_t)(rho0 + lrow), (uint32_t)(((2 * h + kb) * 16 + ((mat & 2) ? 8 : 0)) * 2)));'
assert old in k
new='''if (MW_FUSED) {'''+old+'''} else {
 const int cc=32*(b0+h)+16*kb+2*(lane&3);
 const int ja=jw+16*wiw+(lane>>2),jb=ja+8;
 const auto* z=static_cast<const __nv_bfloat16*>(p.zres);
 rz[0]=(iw<p.N && ja<p.N)?*(const uint32_t*)(z+((size_t)iw*p.N+ja)*CZ+cc):0;
 rz[1]=(iw<p.N && jb<p.N)?*(const uint32_t*)(z+((size_t)iw*p.N+jb)*CZ+cc):0;
 rz[2]=(iw<p.N && ja<p.N)?*(const uint32_t*)(z+((size_t)iw*p.N+ja)*CZ+cc+8):0;
 rz[3]=(iw<p.N && jb<p.N)?*(const uint32_t*)(z+((size_t)iw*p.N+jb)*CZ+cc+8):0;
}'''
k=k.replace(old,new)
head='''// SPDX-License-Identifier: Apache-2.0
// Anthropic f4f62fa K3. Always fused output LN; optional shared x_n input.
// Split-input path reads original residual from global, no training saves.
#include "tmn_kernels.cuh"
namespace tmn { namespace sm90 {
using Cfg=K3Cfg<128,256,0,MW_BI,MW_BJ,MW_SLOT,MW_ACC>;
'''
foot='''}}
extern "C" __global__ __launch_bounds__(384,1)
void infer_k3(__grid_constant__ const tmn::K3Params p){tmn::sm90::infer_k3_body<tmn::sm90::Cfg,1>(p);}
'''
(R/'k3.cu').write_text(head+k+foot)
for fname in ('separate_ln.cu','separate_ln_tma.cu'):
 t=(R.parent/'anthropic_ln_ab_20260919'/fname).read_text()
 t='\n'.join(line for line in t.splitlines() if 'if(q==0)' not in line)
 (R/fname).write_text(t+'\n')
# Same existing Triton arithmetic/tiling, compile out both stat stores for inference.
t=(E/'src/miniworld_engine/kernels/layernorm/triton/main.py').read_text();a=t.index('@triton.jit\ndef layer_norm_fwd_fused');b=t.index('\n@',a+10)
t=t[a:b];t=t[:t.index('\n\n#') ] if '\n\n#' in t else t
# Trim exactly through function body by the next top-level comment; decorators below already excluded.
t='\n'.join(line for line in t.splitlines() if 'tl.store(Mean' not in line and 'tl.store(Rstd' not in line)
(R/'triton_ln.py').write_text('import triton\nimport triton.language as tl\n'+t+'\n')
