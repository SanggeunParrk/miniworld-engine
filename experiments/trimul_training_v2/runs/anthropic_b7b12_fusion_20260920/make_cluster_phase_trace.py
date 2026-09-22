from pathlib import Path
import json
p=Path(__file__).resolve().parent;s=(p/'front_cluster512.cu').read_text();a=s.index('struct Bars');s=s[:a]+'''TMN_DEVI void stamp(const Params& p,int point){if(threadIdx.x==0)asm volatile("{.reg .b32 t;mov.u32 t, %%clock;st.global.u32 [%0],t;}"::"l"(p.counts+2+blockIdx.x*8+point):"memory");}\n'''+s[a:]
s=s.replace('uint8_t* inp=sm+65536+(r%2)*65536;','uint8_t* inp=sm+65536+(r%2)*65536;if(r==8)stamp(p,0);')
s=s.replace('   uint32_t mask=', '   if(r==8)stamp(p,1);uint32_t mask=')
s=s.replace('   int half=wi%2;', '   if(r==8)stamp(p,2);int half=wi%2;')
s=s.replace('   if((r+1)%segment', '   if(r==8)stamp(p,3);if((r+1)%segment')
s=s.replace('int row=tile*64;uint32_t gate_packed[8];', 'int row=tile*64;if(round==2)stamp(p,0);uint32_t gate_packed[8];')
s=s.replace('  for(int side=0;side<2;++side){','  if(round==2)stamp(p,1);for(int side=0;side<2;++side){')
s=s.replace('  // B10 outputs BF16 dx_n.','  if(round==2)stamp(p,2);\n  // B10 outputs BF16 dx_n.')
s=s.replace('  if(threadIdx.x==0){if(tile+CLUSTERS*4<p.tiles)', '  if(round==2)stamp(p,3);if(threadIdx.x==0){if(tile+CLUSTERS*4<p.tiles)')
(p/'front_cluster512_phase_trace.cu').write_text(s);cf=json.loads((p/'front_cluster512.launch.json').read_text());cf['trace_words']=8;(p/'front_cluster512_phase_trace.launch.json').write_text(json.dumps(cf))
(p/'trace_cluster_phases.py').write_text('''from cluster_plan import *
with torch.no_grad():
 a=setup(768);p=ClusterPlan(a,120,'front_cluster512_phase_trace');p();torch.cuda.synchronize();v=p.counts[2:].reshape(120,8).to(torch.int64).cpu();delta=(v[:,1:4]-v[:,:3])&0xffffffff;res={}
 for role in ['dw','dx']:
  vals=delta[(torch.arange(120)%8<4) if role=='dw' else (torch.arange(120)%8>=4)]
  res[role]=dict(median_cycles=vals.median(0).values.tolist(),samples=vals.tolist())
 (R/'cluster512-phase-trace.json').write_text(json.dumps(res,indent=2));print('PHASES',{k:v['median_cycles'] for k,v in res.items()},flush=True)
''')
