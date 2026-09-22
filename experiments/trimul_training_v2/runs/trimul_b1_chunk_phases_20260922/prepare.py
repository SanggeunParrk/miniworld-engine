from pathlib import Path
R=Path(__file__).resolve().parent;S=R.parent/'trimul_b1_wait_folding_20260921'
for p in [S/'replace_plan.py',*S.glob('*.cu'),*S.glob('*.cuh'),*S.glob('*.inc')]:
    (R/p.name).write_text(p.read_text())
p=R/'lowreg_stats.inc';s=p.read_text()
s=s.replace('float (&wp1)[64]){','float (&wp1)[64],int tile_end){').replace('<p.tiles','<tile_end')
p.write_text(s)
p=R/'b1_fused.cu';s=p.read_text()
pos=s.index('TMN_DEVI void shared_role(')
load='''// Inverse of store_weight. Preserve FP32 accumulation across phase chunks.
TMN_DEVI void load_weight(const Params& p,float (&acc)[64],int kind){
 int wi=threadIdx.x/128,lane=threadIdx.x%32,w=(threadIdx.x/32)%4;
 const float* src=p.partw+blockIdx.x*49152+kind*16384;
 #pragma unroll
 for(int q=0;q<16;++q){int rr=w*16+lane/4+(kind==0?wi*64:0),c=q*8+2*(lane%4)+(kind==0?0:wi*128),stride=kind==0?128:256;
  asm volatile("ld.global.v2.f32 {%0,%1},[%2];":"=f"(acc[4*q]),"=f"(acc[4*q+1]):"l"(src+rr*stride+c):"memory");
  asm volatile("ld.global.v2.f32 {%0,%1},[%2];":"=f"(acc[4*q+2]),"=f"(acc[4*q+3]):"l"(src+(rr+8)*stride+c):"memory");
 }
}
'''
s=s[:pos]+load+s[pos:]
a=s.index('TMN_DEVI void shared_role(');b=s.index('\n#if GATE_PHASE\nTMN_DEVI void load_gate_operands',a)
part=s[a:b].replace('uint64_t* bars){','uint64_t* bars,int tile_begin,int tile_end){',1)
part=part.replace('fragment_mask<UCOUNT>(p,blockIdx.x)','fragment_mask<UCOUNT>(p,tile_begin+blockIdx.x)')
part=part.replace('float wp0[64]={},wp1[64]={};','float wp0[64]={},wp1[64]={};\n if(tile_begin){load_weight(p,wp0,1);load_weight(p,wp1,2);}')
part=part.replace('if(blockIdx.x<p.tiles)','if(tile_begin+blockIdx.x<tile_end)').replace('blockIdx.x*64','(tile_begin+blockIdx.x)*64')
part=part.replace('tile=blockIdx.x','tile=tile_begin+blockIdx.x').replace('<p.tiles','<tile_end').replace('round>0','(round>0||tile_begin>0)')
part=part.replace('slot,wp0,wp1);','slot,wp0,wp1,tile_end);')
s=s[:a]+part+s[b:]
a=s.index('TMN_DEVI void gate_phase(');b=s.index('\n#endif\nTMN_DEVI void reduce_at',a)
part=s[a:b].replace('uint64_t* bars){','uint64_t* bars,int tile_begin,int tile_end){',1)
part=part.replace('float acc[64]={};','float acc[64]={};\n if(tile_begin)load_weight(p,acc,0);')
part=part.replace('if(blockIdx.x<p.tiles)','if(tile_begin+blockIdx.x<tile_end)').replace('blockIdx.x*64','(tile_begin+blockIdx.x)*64')
part=part.replace('tile=blockIdx.x','tile=tile_begin+blockIdx.x').replace('<p.tiles','<tile_end').replace('round>0','(round>0||tile_begin>0)')
s=s[:a]+part+s[b:]
a=s.index(' shared_role(p,sm,bars);');b=s.index('\n#if PART_ONLY==2',a)
s=s[:a]+'''
 static_assert(!PRODUCER && GATE_PHASE && B1_LOCAL_GATE,"chunk schedule uses two compute warp-groups and local gate rows");
 const int step=B1_PHASE_CHUNK?UCOUNT*B1_PHASE_CHUNK:p.tiles;
 for(int first=0;first<p.tiles;first+=step){
  const int last=min(first+step,p.tiles);
  // Each previous phase waited all its TMA loads/stores before this barrier.
  // Reinitialize both raw-slot parities and the dy parity before a new phase A.
  allsync();if(threadIdx.x==0){for(int b=0;b<3;++b)mbar_init(bars+b,1);fence_barrier_init();}allsync();
  shared_role(p,sm,bars,first,last);
  __threadfence();allsync();
  if(threadIdx.x==0){for(int b=0;b<2;++b)mbar_init(bars+b,1);fence_barrier_init();}allsync();
  gate_phase(p,sm,bars,first,last);
  __threadfence();allsync();
 }
'''+s[b:]
p.write_text(s)
H=R.parent/'trimul_b1_param_hybrid_20260922'
t=(H/'tune.py').read_text().replace('B1_PARAM_HYBRID','B1_PHASE_CHUNK').replace("default='0,1,2,3'","default='0,2,4,8,16'")
# Always restore mutated inputs before moving to the next candidate.
t=t.replace("row=dict(config=cfg,host=platform.node(),checks=[])","row=dict(config=cfg,host=platform.node(),checks=[]);snap=[];tensors=[]")
t=t.replace("except Exception as e:row.update(valid=False,error=str(e)[-1800:])", "except Exception as e:row.update(valid=False,error=str(e)[-1800:])\n  finally:\n   for x,v in zip(tensors,snap):x.copy_(v)")
(R/'tune.py').write_text(t)
(R/'tune.sbatch').write_text((H/'tune.sbatch').read_text().replace(H.name,R.name).replace('b1-param-hybrid','b1-chunk-phases'))
