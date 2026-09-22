from pathlib import Path
p=Path(__file__).resolve().parent;s=(p/'front_prefetch_lnpair.cu').read_text()
for level in ['L1','L2']:
 for which in ['dw','both']:
  v=s
  instr='asm volatile("prefetch.global.%s [%%0];"::"l"(p.mask+row):"memory");'%level
  a=v.index('TMN_DEVI void load_dw');b=v.index('TMN_DEVI void weight_role',a);f=v[a:b];f=f.replace('if(threadIdx.x)return;','if(threadIdx.x)return;'+instr);v=v[:a]+f+v[b:]
  if which=='both':
   a=v.index('TMN_DEVI void issue_gate');b=v.index('TMN_DEVI void load_ln_next',a);f=v[a:b].replace('if(threadIdx.x)return;','if(threadIdx.x)return;'+instr);v=v[:a]+f+v[b:]
  name='front_prefetch_lnpair_mask'+level+'_'+which;(p/(name+'.cu')).write_text(v);(p/(name+'.launch.json')).write_text((p/'front_kindprefetch.launch.json').read_text())
