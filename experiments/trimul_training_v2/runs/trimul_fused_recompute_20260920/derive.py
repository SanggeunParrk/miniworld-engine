"""Generate new fused backward sources; originals remain unchanged."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
P7=R.parent/'anthropic_b7b12_fusion_20260920'
P1=R.parent/'anthropic_b1b4_pipeline_20260919'
s=(P7/'front_prefetch_lnpair_storepipe.cu').read_text()
s=s.replace('#include "warp_primitives.cuh"','#include "common_recompute.cuh"')
s=s.replace('#define WGRAD_SLICES 2','#ifndef WGRAD_SLICES\n#define WGRAD_SLICES 2\n#endif')
s=s.replace('segmentRounds=(rounds+1)/2','segmentRounds=(rounds+WGRAD_SLICES-1)/WGRAD_SLICES')
s=s.replace('(group*DW_SPLITS+split)*2+r/segmentRounds','(group*DW_SPLITS+split)*WGRAD_SLICES+r/segmentRounds')
s=s.replace('DW_SPLITS*2','DW_SPLITS*WGRAD_SLICES')
s=s.replace('gl=pair_get(s,c*2,r),pr=pair_get(s,c*2+1,r)','gl=pair_get(s,c,r),pr=pair_get(s+8192,c,r)')
s=s.replace('int M,L,tiles;','int M,L,tiles;const float* beta;')
start=s.index('TMN_DEVI void load_dw(');end=s.index('\nTMN_DEVI void weight_role',start)
s=s[:start]+(R/'b7_recompute_helpers.inc').read_text()+s[end:]
s=s.replace('glu_small(p,s,s+40960,s+49152,tile*64);','recompute_dw(p,s,gamma,beta);glu_small(p,s,s+40960,s+49152,tile*64);')
s=s.replace('void weight_role(const Params& p,uint8_t* sm,uint64_t* b){','void weight_role(const Params& p,uint8_t* sm,uint64_t* b,const float* gamma,const float* beta){')
start=s.index('TMN_DEVI void load_g(');end=s.index('\nTMN_DEVI void load_p',start)
s=s[:start]+'''TMN_DEVI void load_g(const Params& p,uint8_t* sm,uint64_t* b,int row,int side,int h){
 if(threadIdx.x)return;int slot=h&1;uint8_t* s=sm+slot*DX_SLOT;mbar_arrive_expect_tx(b+slot,40960);
 tma_load_2d(s+16384,side?&p.dr:&p.dl,b+slot,row,h*64);
 for(int n=0;n<2;++n){
  tma_load_2d(s+n*8192,side?&p.wr:&p.wl,b+slot,h*64,n*64);
  tma_load_2d(s+24576+n*8192,side?&p.wrg:&p.wlg,b+slot,h*64,n*64);
 }
}
'''+s[end:]
s=s.replace('const float* gamma){\n int split=blockIdx.x-DWCOUNT','const float* gamma,const float* beta){\n int split=blockIdx.x-DWCOUNT')
s=s.replace('  allsync();\n  if(threadIdx.x==0&&round>0)',
'''  allsync();
  if(threadIdx.x==0){mbar_arrive_expect_tx(bar+4,16384);for(int c=0;c<2;++c)tma_load_2d(sm+114688+c*8192,&p.x,bar+4,c*64,row);}
  mbar_wait(bar+4,round&1);
  if(threadIdx.x<128)normalize_tile<8>(sm+114688,sm+114688,gamma,beta);
  fence_proxy_async();allsync();
  if(threadIdx.x==0&&round>0)''')
s=s.replace('mbar_wait(bar+slot,h/2);glu_small(p,s,s+16384',
            'mbar_wait(bar+slot,h/2);recompute_pre(s,sm+114688,s+24576,s);glu_small(p,s,s+16384')
s=s.replace('float mu[2]={p.mean[row+ra],p.mean[row+rb]},rs[2]={p.rs[row+ra],p.rs[row+rb]},s1[2]={},s2[2]={};',
'''LnStats recstats=normalize_tile<8,false,false,false>(lnsm,nullptr,gamma,beta);
  float mu[2]={recstats.mA,recstats.mB},rs[2]={recstats.rA,recstats.rB},s1[2]={},s2[2]={};''')
s=s.replace('__launch_bounds__(256,2)','__launch_bounds__(256,1)')
s=s.replace('__shared__ uint64_t bar[4];__shared__ float gamma[128];if(threadIdx.x<128)gamma[threadIdx.x]=p.gamma[threadIdx.x];',
'''__shared__ uint64_t bar[5];__shared__ float gamma[128],beta[128];if(threadIdx.x<128){gamma[threadIdx.x]=p.gamma[threadIdx.x];beta[threadIdx.x]=p.beta[threadIdx.x];}''')
s=s.replace('for(int i=0;i<4;++i)mbar_init','for(int i=0;i<5;++i)mbar_init')
s=s.replace('weight_role(p,sm,bar);else input_role(p,sm,bar,gamma);','weight_role(p,sm,bar,gamma,beta);else input_role(p,sm,bar,gamma,beta);')
(R/'b7_fused.cu').write_text('// On-chip full recomputation; no global xn/pre/LN-stat reads.\n'+s)

s1=(P1/'dual_ln_prefetch.cu').read_text()
a=s1.index('struct MaskCycle');b=s1.index('// DW: two 96',a)
helpers=s1[a:b]
# Reuse unchanged gate derivatives and register-local output-LN backward.
# B1's new role scheduler prepares the same on-chip layouts.
helpers=helpers.replace('int split=blockIdx.x;', 'int split=blockIdx.x;')
(R/'b1_saved_math.inc').write_text(helpers)
(R/'derivation.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
 (P7/'front_prefetch_lnpair_storepipe.cu',P1/'dual_ln_prefetch.cu',R/'common_recompute.cuh',R/'b7_recompute_helpers.inc')},indent=2))
