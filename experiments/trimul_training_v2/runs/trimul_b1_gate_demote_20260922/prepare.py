from pathlib import Path
R=Path(__file__).resolve().parent;S=R.parent/'trimul_b1_tma_cache_priority_20260922'
for p in [S/'replace_plan.py',*S.glob('*.cu'),*S.glob('*.cuh'),*S.glob('*.inc')]:
    (R/p.name).write_text(p.read_text())
p=R/'b1_fused.cu';s=p.read_text();a=s.index('TMN_DEVI void load_gate_operands(');b=s.index('TMN_DEVI void gate_phase(',a)
part=s[a:b]
part=part.replace('tma_load_2d(dst,&p.dgmap,','cached_load<(B1_GATE_DEMOTE&1)?-1:0>(dst,&p.dgmap,')
part=part.replace('tma_load_2d(dst+8192,&p.dgmap,','cached_load<(B1_GATE_DEMOTE&1)?-1:0>(dst+8192,&p.dgmap,')
part=part.replace('tma_load_2d(dst+16384,&p.x,','cached_load<(B1_GATE_DEMOTE&2)?-1:0>(dst+16384,&p.x,')
part=part.replace('tma_load_2d(dst+24576,&p.x,','cached_load<(B1_GATE_DEMOTE&2)?-1:0>(dst+24576,&p.x,')
s=s[:a]+part+s[b:];p.write_text(s)
t=(S/'tune.py').read_text().replace("default='0,1,2,3,4,7,8,11,15'","default='0,1,2,3'")
t=t.replace("cfg['defines']['B1_TMA_PRIORITY']=level;","cfg['defines']['B1_TMA_PRIORITY']=2;cfg['defines']['B1_GATE_DEMOTE']=level;")
(R/'tune.py').write_text(t)
(R/'tune.sbatch').write_text((S/'tune.sbatch').read_text().replace(S.name,R.name).replace('b1-tma-priority','b1-gate-demote'))
