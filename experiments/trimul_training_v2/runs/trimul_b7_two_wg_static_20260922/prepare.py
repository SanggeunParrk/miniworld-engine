from pathlib import Path
R=Path(__file__).resolve().parent;S=R.parent/'trimul_b7_two_wg_lowreg_20260922'
for name in ('plan.py','single_wg.inc','pair_producer.inc','precompile.py','probe.py','ncu_profile.py','sanitize.py','run.slurm','sanitize.slurm','profile.slurm','sweep.py'):
 (R/name).write_text((S/name).read_text().replace(S.name,R.name))
s=(S/'joint.cu').read_text().replace('TMN_DEVI void consumer_compute(', 'template<int G> TMN_DEVI void consumer_compute(').replace('int g=threadIdx.x/128-1,rank=', 'constexpr int g=G;int rank=')
old='else{setmaxnreg_inc<104>();consumer_compute(p,sm,bar,gamma,beta);}'
new='else if(wi==1){setmaxnreg_inc<104>();consumer_compute<0>(p,sm,bar,gamma,beta);}else{setmaxnreg_inc<104>();consumer_compute<1>(p,sm,bar,gamma,beta);}'
assert old in s;s=s.replace(old,new);(R/'joint.cu').write_text(s)
