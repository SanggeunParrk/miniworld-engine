from pathlib import Path
R=Path(__file__).resolve().parent;S=R.parent/'trimul_b1_wait_folding_20260921'
for p in [S/'replace_plan.py',*S.glob('*.cu'),*S.glob('*.cuh'),*S.glob('*.inc')]:
    (R/p.name).write_text(p.read_text())
p=R/'b1_fused.cu';s=p.read_text()
s=s.replace('TMN_DEVI void shared_role(const Params& p,uint8_t* sm,uint64_t* bars){',
'''TMN_DEVI void shared_role(const Params& p,uint8_t* sm,uint64_t* bars,float (&wp0)[64],float (&wp1)[64]){''')
assert ' float wp0[64]={},wp1[64]={};' in s
s=s.replace(' float wp0[64]={},wp1[64]={};','')
old=' store_weight(p,wp0,1);store_weight(p,wp1,2);'
new='''
#if !(B1_LATE_WEIGHT & 1)
 store_weight(p,wp0,1);
#endif
#if !(B1_LATE_WEIGHT & 2)
 store_weight(p,wp1,2);
#endif'''
assert old in s;s=s.replace(old,new)
s=s.replace(' shared_role(p,sm,bars);',' float wp0[64]={},wp1[64]={};\n shared_role(p,sm,bars,wp0,wp1);')
s=s.replace('\n#if PART_ONLY==2\n','''
 // Delay selected partial writes until the gate phase has reused x_n/dGate.
#if B1_LATE_WEIGHT & 1
 store_weight(p,wp0,1);
#endif
#if B1_LATE_WEIGHT & 2
 store_weight(p,wp1,2);
#endif
#if PART_ONLY==2
''')
p.write_text(s)
H=R.parent/'trimul_b1_chunk_phases_20260922'
(R/'tune.py').write_text((H/'tune.py').read_text().replace('B1_PHASE_CHUNK','B1_LATE_WEIGHT').replace("default='0,2,4,8,16'","default='0,1,2,3'"))
(R/'tune.sbatch').write_text((H/'tune.sbatch').read_text().replace(H.name,R.name).replace('b1-chunk-phases','b1-weight-late'))
