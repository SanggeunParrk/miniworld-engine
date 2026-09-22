from pathlib import Path
import sys,subprocess,os
R=Path(__file__).resolve().parent;variant=sys.argv[1] if len(sys.argv)>1 else 'baseline';stem=R/('ncu-'+variant)
env=dict(os.environ);env.pop('PYTHONPATH',None);env['PYTHONNOUSERSITE']='1'
metrics='gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,smsp__inst_executed.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum'
cmd=['ncu','--profile-from-start','off','--section','SpeedOfLight','--section','SchedulerStats','--section','SourceCounters','--metrics',metrics,'--cache-control','none','--clock-control','none','--target-processes','all','--kernel-name','regex:^(transition_bwd_fused|reduce_partials)$','--launch-count','2','--force-overwrite','-o',str(stem),'env','PYTHONPATH='+os.environ.get('PYTHONPATH',''),sys.executable,'-B',str(R/'profile_launch.py'),'--variant',variant]
with stem.with_suffix('.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
with stem.with_suffix('.csv').open('w') as f:subprocess.run(['ncu','--import',str(stem)+'.ncu-rep','--page','raw','--csv'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
print('DONE',variant,flush=True)
